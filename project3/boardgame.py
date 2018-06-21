# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
#
# Revision history
# Originally written in Matlab by Sae-Young Chung in Apr. 2016
#   for EE405C Electronics Design Lab <Network of Smart Systems>, Spring 2016
# Python & TensorFlow porting by Wonseok Jeon, Narae Ryu and Jinhak Kim,
#                                    Hwehee Chung, Sungik Choi in Nov. 2016
#   for EE488C Special Topics in EE <Deep Learning and AlphaGo>, Fall 2016
# Revised by Sae-Young Chung on 2017/12/05
#   for EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017

import numpy as np
import tensorflow as tf
from scipy.signal import convolve2d as conv2d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import tkMessageBox
import time

def data_augmentation(d, w, pos):
    # data augmentation
    # original, horizontal flip, vertical flip, and both
    # if the board is square, additionally 90 degree rotation
    [nx, ny, nc, ng] = d.shape
    if nx == ny:
        f = 8
    else:
        f = 4
    ng_new = ng * f
    dnew = np.zeros((nx, ny, nc, ng_new))
    wnew = np.zeros((ng_new))
    posnew = np.zeros((nx, ny, ng_new))
    dnew[:, :, :, np.arange(ng)] = d[:, :, :, :]
    dnew[:, :, :, np.arange(ng) + 1 * ng] = d[::-1, :, :, :]
    dnew[:, :, :, np.arange(ng) + 2 * ng] = d[:, ::-1, :, :]
    dnew[:, :, :, np.arange(ng) + 3 * ng] = d[::-1, ::-1, :, :]
    if np.sum(pos):
        posnew[:, :, np.arange(ng)] = pos[:, :, :]
        posnew[:, :, np.arange(ng) + 1 * ng] = pos[::-1, :, :]
        posnew[:, :, np.arange(ng) + 2 * ng] = pos[:, ::-1, :]
        posnew[:, :, np.arange(ng) + 3 * ng] = pos[::-1, ::-1, :]

    if f==8:
        dnew[:, :, :, np.arange(ng) + 4 * ng] = np.rollaxis(d, 1, 0)
        dnew[:, :, :, np.arange(ng) + 5 * ng] = dnew[::-1, :, :, np.arange(ng) + 4 * ng]
        dnew[:, :, :, np.arange(ng) + 6 * ng] = dnew[:, ::-1, :, np.arange(ng) + 4 * ng]
        dnew[:, :, :, np.arange(ng) + 7 * ng] =\
                dnew[::-1, ::-1, :, np.arange(ng) + 4 * ng]
        if np.sum(pos):
            posnew[:, :, np.arange(ng) + 4 * ng] = np.rollaxis(pos, 1, 0)
            posnew[:, :, np.arange(ng) + 5 * ng] = posnew[::-1, :, np.arange(ng) + 4 * ng]
            posnew[:, :, np.arange(ng) + 6 * ng] = posnew[:, ::-1, np.arange(ng) + 4 * ng]
            posnew[:, :, np.arange(ng) + 7 * ng] =\
                    posnew[::-1, ::-1, np.arange(ng) + 4 * ng]

    for k in range(f):
        wnew[k * ng + np.arange(ng)]=w

    return dnew, wnew, posnew

class board_game(object):
    def next_move(self, b, state, game_in_progress, net, rn, p, move, nlevels = 1, rw = 0):
        # returns next move by using neural networks
        # this is a parallel version, i.e., returns next moves for multiple games
        # Input arguments: b,state,game_in_progress,net,rn,p,move,nlevels,rw
        #   b: current board states for multiple games
        #   state: extra states
        #   game_in_progress: 1 if game is in progress, 0 if ended
        #   net: neural network. can be empty (in that case 'rn' should be 1)
        #   rn: if 0 <= rn <= 1, controls randomness in each move (0: no randomness, 1: pure random)
        #     if rn = -1, -2, ..., then the first |rn| moves are random
        #   p: current player (1: black, 2: white)
        #   move: k-th move (1,2,3,...)
        #   nlevels (optional): tree search depth (1,2, or 3). default=1
        #     if nlevels is even, then 'net' should be the opponent's neural network
        #   rw (optional): randomization in calculating winning probabilities, default=0
        # Return values
        # new_board,new_state,valid_moves,wp_max,wp_all,x,y=next_move(b,game_in_progress,net,rn,p,move)
        #   new_board: updated board states containing new moves
        #   new_state: updated extra states
        #   n_valid_moves: number of valid moves
        #   wp_max: best likelihood of winning
        #   wp_all: likelihood of winning for all possible next moves
        #   x: x coordinates of the next moves in 'new_board'
        #   y: y coordinates of the next moves in 'new_board'
        
        # board size
        nx = self.nx; ny = self.ny; nxy = nx * ny
        # randomness for each game & minimum r
        r = rn; rmin = np.amin(r)
        # number of games
        if b.ndim>=3:
            ng = b.shape[2]
        else:
            ng=1
        # number of valid moves in each game 
        n_valid_moves = np.zeros((ng))
        # check whether each of up to 'nxy' moves is valid for each game
        valid_moves = np.zeros((ng, nxy))
        # win probability for each next move
        wp_all = np.zeros((nx, ny, ng))
        # maximum of wp_all over all possible next moves
        wp_max = -np.ones((ng))
        mx = np.zeros((ng))
        my = np.zeros((ng))
        x = -np.ones((ng))
        y = -np.ones((ng))

        # check nlevels
        if nlevels > 3 or nlevels <= 0:
            raise Exception('# of levels not supported. Should be 1, 2, or 3.')
        # total cases to consider in tree search
        ncases = pow(nxy, nlevels)

        # maximum possible board states considering 'ncases'
        d = np.zeros((nx, ny, 3, ng * ncases), dtype = np.int32)

        for p1 in range(nxy):
            vm1, b1, state1 = self.valid(b, state, self.xy(p1), p)
            n_valid_moves += vm1
            if rmin < 1:
                valid_moves[:, p1] = vm1
                if nlevels == 1:
                    c = 3 - p  # current player is changed to the next player after placing a stone at 'p1'
                    idx = np.arange(ng) + p1 * ng
                    d[:, :, 0, idx] = (b1 == c)     # 1 if current player's stone is present, 0 otherwise
                    d[:, :, 1, idx] = (b1 == 3 - c) # 1 if opponent's stone is present, 0 otherwise
                    d[:, :, 2, idx] = 2 - c         # 1: current player is black, 0: white
                else:
                    for p2 in range(nxy):
                        vm2, b2, state2 = self.valid(b1, state1, self.xy(p2), 3 - p)
                        if nlevels == 2:
                            c = p                 # current player is changed again after placing a stone at 'p2'
                            idx = np.arange((ng)) + p1 * ng + p2 * ng * nxy
                            d[:, :, 0, idx] = (b2 == c)
                            d[:, :, 1, idx] = (b2 == 3 - c)
                            d[:, :, 2, idx] = 2 - c
                        else:
                            for p3 in range(nxy):
                                vm3, b3, state3 = self.valid(b2, state2, self.xy(p3), p)
                                c = 3 - p         # current player is changed yet again after placing a stone at 'p3'
                                idx = np.arange(ng) + p1 * ng + p2 * ng * nxy\
                                        + p3 * ng * nxy * nxy
                                d[:, :, 0, idx] = (b3 == c)
                                d[:, :, 1, idx] = (b3 == 3 - c)
                                d[:, :, 2, idx] = 2 - c

        # n_valid_moves is 0 if game is not in progress
        n_valid_moves = n_valid_moves * game_in_progress

        # For operations in TensorFlow, load session and graph
        sess = tf.get_default_session()

        # d(nx, ny, 3, ng * ncases) becomes d(ng * ncases, nx, ny, 3)
        d = np.rollaxis(d, 3)
        if rmin < 1: # if not fully random, then use the neural network 'net'
            softout = np.zeros((d.shape[0], 3))
            size_minibatch = 1024
            num_batch = np.ceil(d.shape[0] / float(size_minibatch))
            for batch_index in range(int(num_batch)):
                batch_start = batch_index * size_minibatch
                batch_end = \
                        min((batch_index + 1) * size_minibatch, d.shape[0])
                indices = range(batch_start, batch_end)
                feed_dict = {'S:0': d[indices, :, :, :]}  # d[indices,:,:,:] goes to 'S' (neural network input)
                softout[indices, :] = sess.run(net, feed_dict = feed_dict) # get softmax output from 'net'
            if p == 1:   # if the current player is black
                # softout[:,0] is the softmax output for 'tie'
                # softout[:,1] is the softmax output for 'black win'
                # softout[:,2] is the softmax output for 'white win'
                wp = 0.5 * (1 + softout[:, 1] - softout[:, 2])  # estimated win prob. for black
            else:        # if the current player is white
                wp = 0.5 * (1 + softout[:, 2] - softout[:, 1])  # estimated win prob. for white

            if rw != 0:     # this is only for nlevels == 1
                # add randomness so that greedy action selection to be done later is randomized
                wp = wp + np.random.rand((ng, 1)) * rw

            if nlevels >= 3:
                wp = np.reshape(wp, (ng, nxy, nxy, nxy))
                wp = np.amax(wp, axis = 3)    

            if nlevels >= 2:
                wp = np.reshape(wp, (ng, nxy, nxy))
                wp = np.amin(wp, axis = 2)

            wp = np.transpose(np.reshape(wp,(nxy,ng)))
            wp = valid_moves * wp - (1 - valid_moves)
            wp_i = np.argmax(wp, axis = 1)  # greedy action selection
            mxy = self.xy(wp_i)             # convert to (x,y) coordinates

            for p1 in range(nxy):
                pxy = self.xy(p1)
                wp_all[int(pxy[:, 0]), int(pxy[:, 1]), :] = wp[:, p1]  # win prob. for each of possible next moves

        new_board = np.zeros(b.shape)
        new_board[:, :, :] = b[:, :, :]
        new_state = np.zeros(state.shape)
        new_state[:, :] = state[:, :]

        for k in range(ng):
            if n_valid_moves[k]: # if there are valid moves
                if (r[k] < 0 and np.ceil(move / 2.) <= -r[k])\
                        or (r[k] >= 0 and np.random.rand() <= r[k]):
                    # if r[k]<0, then randomize the next move if # of moves is <= |r[k]|
                    # if 0<r[k]<=1, then randomize the next move with probability r[k]
                    # randomization is uniform over all possible valid moves
                    while True:
                        # random position selection
                        rj = np.random.randint(nx)
                        rk = np.random.randint(ny)
                        rxy = np.array([[rj, rk]])
                        isvalid, _, _ = self.valid(b[:, :, [k]], state[:, [k]], rxy, p)
                        if int(isvalid[0]):
                            break

                    isvalid, bn, sn = self.valid(b[:, :, [k]], state[:, [k]], rxy, p)
                    new_board[:, :, [k]] = bn
                    new_state[:, [k]] = sn
                    x[k] = rj
                    y[k] = rk

                else:
                    isvalid, bn, sn = self.valid(b[:, :, [k]], state[:, [k]], mxy[[k], :], p)
                    new_board[:, :, [k]] = bn
                    new_state[:, [k]] = sn
                    x[k] = mxy[k, 0]
                    y[k] = mxy[k, 1]

            else: # if there is no more valid move
                isvalid, bn, sn = self.valid(b[:, :, [k]], state[:, [k]], -np.ones((1, 2)), p)
                new_state[:, [k]] = sn

        return new_board, new_state, n_valid_moves, wp_max, wp_all, x, y


    def play_games(self, net1, r1, net2, r2, ng, max_time = 0, nargout = 1):
        # plays 'ng' games between two players
        # Inputs
        #   net1: neural network playing black. can be empty (r1 should be 1 if net1 is empty)
        #   r1: if 0 <= r1 <= 1, controls randomness in the next move by player 1 (0: no randomness, 1: pure random)
        #     if r1 = -1, -2, ..., then the first |r1| moves are random
        #   net2: neural network playing white. can be empty (r2 should be 1 if net2 is empty)
        #   r2: if 0 <= r2 <= 1, controls randomness in the next move by player 2 (0: no randomness, 1: pure random)
        #     if r2 = -1, -2, ..., then the first |r2| moves are random
        #   ng: number of games to play
        #   max_time (optional): the max. number of moves per game
        #   nargout (optional): the number of output arguments
        # Return values
        #   stat=play_games(net1,r1,net2,r2,ng,nargout=1): statistics for net1, stat=[win loss tie]
        #   d,w,wp,stat=play_games(net1,r1,net2,r2,ng,nargout=2,3, or 4)
        #     d: 4-d tensor of size nx*ny*3*nb containing all moves, where nb is the total number of board states
        #     w: nb*1, 0: tie, 1: black wins, 2: white wins
        #     wp (if nargout>=3):  win probabilities for the current player
        #     stat (if nargout==4): statistics for net1, stat=[win loss tie], for net2, swap win & loss
        
        
        # board size 
        nx = self.nx; ny = self.ny

        # maximum number of moves in each game
        if max_time <= 0:
            np0 = nx * ny * 2
        else:
            np0 = max_time

        # m: max. possible number of board states
        m = np0 * ng
        d = np.zeros((nx, ny, 3, m))
        
        # game outcome, tie(0)/black win(1)/white win(2), for each board state
        w = np.zeros((m))

        # winning probability       
        wp = np.zeros((m))

        # 1 means valid as training data, 0 means invalid
        valid_data = np.zeros((m))

        # current turn: 1 if black, 2 if white
        turn = np.zeros((m))

        # number of valid moves in the previous move
        vm0 = np.ones((ng))

        # initialize game
        if hasattr(self, 'game_init'): 
            [b, state] = self.game_init(ng)
        else:   # default initialization
            b = np.zeros((nx, ny, ng))
            state = np.zeros((0, ng))

        # maximum winning probability for each game
        wp_max = np.zeros((ng))

        # 1 if game is in progress, 0 otherwise
        game_in_progress = np.ones((ng))

        # first player is black (1)
        p = 1

        for k in range(np0):
            if p == 1:   # if black's turn, use net1 and r1
                b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                    self.next_move(b, state, game_in_progress, net1, r1, p, k)
            else:        # if white's turn, use net2 and r2
                b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                    self.next_move(b, state, game_in_progress, net2, r2, p, k)

            # check if game ended and if winner is decided
            w0, end_game, _, _ = self.winner(b, state)
            idx = np.arange(k * ng, (k + 1) * ng)
            c = 3 - p    # current player is now changed to the next player
            d[:, :, 0, idx] = (b == c)      # 1 if current player's stone is present, 0 otherwise
            d[:, :, 1, idx] = (b == 3 - c)  # 1 if opponent's stone is present, 0 otherwise
            d[:, :, 2, idx] = 2 - c         # color of current player's stone (1 if black, 0 if white)
            
            wp[idx] = wp_max
            # valid as training data if game is in progress and if # of valid moves is > 0
            valid_data[idx] = game_in_progress * (n_valid_moves > 0)
            
            # information on the current player
            turn[idx] = p
            
            # update game_in_progress
            game_in_progress *=\
                    ((n_valid_moves > 0) * (end_game == 0) +\
                    ((vm0 + n_valid_moves) > 0) * (end_game == -1))
            # if end_game==1, game ends
            # if end_game==0, game ends if no more move is possible for the current player
            # if end_game==-1, game ends if no moves are possible for both players

            number_of_games_in_progress = np.sum(game_in_progress)
            if number_of_games_in_progress == 0:
                break   # no games to play

            p = 3 - p   # change the turn
            vm0 = n_valid_moves[:]  # preserve 'n_valid_moves'
                                    # no copying, which is ok since 'n_valid_moves' will be created as
                                    # a new array in the next step

        for k in range(np0):
            idx = np.arange(k * ng, (k + 1) * ng)
            w[idx] = w0[:] # final winner

        # player 1's stat
        win = np.sum(w0 == 1) / float(ng)
        loss = np.sum(w0 == 2) / float(ng)
        tie = np.sum(w0 == 0) / float(ng)

        varargout = []

        if nargout >= 2:
            fv = np.where(valid_data)[0]
            varargout.append(d[:, :, :, fv])
            varargout.append(w[fv])
            if nargout >= 3:
                varargout.append(wp[fv])
            if nargout >= 4:
                varargout.append([win, loss, tie])
        else:
            varargout.append([win, loss, tie])
        return varargout
 
    def play_interactive(self, net1, r1, net2, r2):
        # interactive board game
        # Usage 1)
        # game.play_interactive([],0,[],0): human vs human
        # Usage 2)
        # game.play_interactive(net1,r1,[],0): computer vs human
        #   net1: neural network, can not be empty (r1 should be 1 if net1 is empty)
        #   r1: randomness for net1
        # Usage 3)
        # game.play_interactive(net1,r1,net2,r2): computer vs computer
        #   (one move at a time when mouse is clicked)
        #   net1: first neural network, can be empty (r1 should be 1 if net1 is empty)
        #   r1: randomness for net1
        #   net2: second neural network, can be empty (r2 should be 1 if net2 is empty)
        #   r2: randomness for net2

        nx = self.nx
        ny = self.ny
        self.show_p = 1

        self.net1 = net1
        self.net2 = net2
        self.r1 = r1
        self.r2 = r2

        if net1 == [] and net2 == []:
            self.human_players = 2
            self.r1 = 1
            self.r2 = 1
        elif net1 != [] and net2 == []:
            self.human_players = 1
            self.r2 = 1
        elif net1 != [] and net2 != []:
            self.human_players = 0
        else:
            self.human_players = 1
            self.net1 = net2
            self.net2 = []
            self.r2 = 1

        self.b = np.zeros((nx, ny, 1))
        self.game_in_progress = 0
        self.move = 0
        self.vm0 = 1
        self.stone = np.zeros((nx, ny), dtype = np.object)
        self.txt = np.zeros((nx, ny), dtype = np.object)
        self.txt_winner = []
        self.board_type = 'go'
        if hasattr(self, 'theme'):
            if self.theme == 'check':
                self.board_type = 'check'
            elif self.theme == 'basic':
                self.board_type = 'basic'
                if min([nx,ny]) <= 3:
                    self.board_color = 'white'
                else:
                    self.board_color = [43.0/255, 123.0/255, 47.0/255]       

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, nx + 0.5)
        ax.set_ylim(0, ny + 0.5)
        self.draw_newboard(ax)
        cid = fig.canvas.mpl_connect('button_press_event', self.mouse_click)
        axexit = plt.axes([0.7,0.03,0.15,0.06])
        axwhite = plt.axes([0.4,0.03,0.15,0.06])
        axblack = plt.axes([0.1,0.03,0.15,0.06])
        ax.axis('equal')
        ax.axis('off')
        bexit = Button(axexit,'Exit')
        bexit.on_clicked(self.push_callback)
        bwhite = Button(axwhite, 'New (white)')
        bwhite.on_clicked(self.push_callback2)
        bblack = Button(axblack, 'New (black)')
        bblack.on_clicked(self.push_callback3)
        plt.show()

    def push_callback(self, event):
        plt.close()

    def push_callback2(self, event):
        self.init_board()
        self.b=np.zeros((self.nx,self.ny,1))
        self.turn=1
        self.my_color=2
        self.game_in_progress=1
        self.move=1
        self.vm0=1
        if hasattr(self, 'game_init'):
            self.b,self.state=self.game_init(1)
            self.draw_board()
        else:
            self.b=np.zeros((self.nx,self.ny,1))
            self.state=np.zeros((0,1))
        plt.draw()
        if self.human_players == 1:
            self.mouse_click([])    # so that computer plays first for computer-human match
                    
    def push_callback3(self, event):
        self.init_board()
        self.b=np.zeros((self.nx,self.ny,1))
        self.turn=1
        self.my_color=1
        self.game_in_progress=1
        self.move=1
        self.vm0=1
        if hasattr(self, 'game_init'):
            self.b,self.state=self.game_init(1)
            self.draw_board()
        else:
            self.b=np.zeros((self.nx,self.ny,1))
            self.state=np.zeros((0,1))
        plt.draw()

    def mouse_click(self, event):
        if event == []:
            xm = 0
            ym = 0
        elif event.button == 1:
            xc = event.xdata
            yc = event.ydata
            if xc > 0.5 and xc < (self.nx + 0.5) and yc > 0.5 and yc < self.ny + 0.5:
                xm = round(xc)
                ym = round(yc)
            else:
                return

        if self.game_in_progress and (self.human_players==2):
            x=xm
            y=ym
            isvalid,bn,new_state=self.valid(self.b,self.state,self.xy(self.nx*(x-1)+y-1),self.turn)
            if isvalid:
                w,end_game,_,_=self.winner(bn,new_state)
                self.b = bn
                self.draw_board()
                plt.draw()
                self.turn=3-self.turn
                self.move += 1
                self.state=new_state
                if end_game==1:
                    self.game_in_progress=0
                    self.display_result()
                elif self.number_of_valid_moves(self.b,self.state,self.turn)==0:
                    if end_game<0:
                        isvalid,bn,self.state=self.valid(self.b,self.state,np.array([[-1,-1]]),self.turn)
                        if self.turn==1:
                            tkMessageBox.showwarning('Message', 'Black must pass.')
                        else:
                            tkMessageBox.showwarning('Message', 'White must pass.')
                        self.turn=3-self.turn
                        self.move +=1
                        if self.number_of_valid_moves(self.b,self.state,self.turn)==0:
                            self.game_in_progress=0
                            self.display_result()
                    else:
                        self.game_in_progress=0
                        self.display_result()
        elif self.game_in_progress and (self.human_players==1):
            x=xm
            y=ym
            if self.turn==self.my_color:
                isvalid,bn,new_state=self.valid(self.b,self.state,self.xy(self.nx*(x-1)+(y-1)),self.turn)
                if isvalid:
                    self.b = bn
                    self.draw_board()
                    self.turn = 3-self.turn
                    self.move +=1
                    self.state=new_state
                    self.vm0=1
                    self.hide_probabilities()
                    plt.draw()
                    _,end_game,_,_=self.winner(self.b,self.state)
                    if end_game==1:
                        self.game_in_progress=0
                        self.display_result()

            # computer may need to play more than once in case human must pass
            while self.game_in_progress and self.turn!=self.my_color:
                start_time = time.time()
                new_board, new_state, n_valid_moves,wp_max, wp_all, ox, oy =\
                    self.next_move(self.b, self.state, self.game_in_progress, self.net1,\
                    np.array([self.r1]), self.turn, self.move)
                end_time = time.time()
                print('%6.3f sec' % (end_time-start_time))

                w,end_game,_,_=self.winner(self.b,new_board)
                if n_valid_moves:
                    self.b = new_board
                    self.draw_board()
                    if self.r1 == 0:
                        self.show_probabilities(wp_all,ox,oy)
                plt.draw()
                if end_game==1:
                    self.game_in_progress=0
                    self.display_result()
                elif end_game==0 and n_valid_moves==0:
                    self.game_in_progress=0
                    self.display_result()
                elif end_game<0 and n_valid_moves+self.vm0==0:
                    self.game_in_progress=0
                    self.display_result()
                elif n_valid_moves==0:
                    self.turn=3-self.turn
                    self.move=self.move+1
                    self.state=new_state
                    self.vm0=n_valid_moves
                    if self.number_of_valid_moves(self.b,self.state,self.turn)==0:
                        self.game_in_progress=0
                        self.display_result()
                    else:
                        tkMessageBox.showwarning('Message', 'I pass.')
                else:
                    self.turn=3-self.turn
                    self.move +=1
                    self.state=new_state
                    self.vm0=n_valid_moves
                    if self.number_of_valid_moves(self.b,self.state,self.turn)==0:
                        if end_game<0:
                            isvalid,bn,self.state=self.valid(self.b,self.state,np.array([[-1,-1]]),self.turn)
                            tkMessageBox.showwarning('Message', 'You must pass.')
                            self.turn = 3-self.turn
                            self.move +=1
                            self.vm0=0
                        else:
                            self.game_in_progress=0
                            self.display_result()
        elif self.game_in_progress:
            if self.turn==self.my_color:
                start_time = time.time()
                new_board, new_state ,n_valid_moves,wp_max, wp_all, ox, oy =\
                    self.next_move(self.b, self.state, self.game_in_progress, self.net1,\
                    np.array([self.r1]), self.turn, self.move)
                end_time = time.time()
                print('%6.3f sec' % (end_time-start_time))
            else:
                start_time = time.time()
                new_board, new_state ,n_valid_moves,wp_max, wp_all, ox, oy =\
                    self.next_move(self.b, self.state, self.game_in_progress, self.net2,\
                    np.array([self.r2]), self.turn, self.move)
                end_time = time.time()
                print('%6.3f sec' % (end_time-start_time))
            w,end_game,_,_=self.winner(new_board,new_state)
            if n_valid_moves:
                self.b = new_board
                self.draw_board()
                if self.turn==self.my_color:
                    if (self.r1==0):
                        self.show_probabilities(wp_all,ox,oy)
                else:
                    if (self.r2==0):
                        self.show_probabilities(wp_all,ox,oy)
                plt.draw()
            self.turn=3-self.turn
            self.move +=1
            self.state=new_state
            if end_game==1:
                self.game_in_progress=0
                self.display_result()
            elif self.number_of_valid_moves(self.b,self.state,self.turn)==0:
                if end_game<0:
                    isvalid,bn,self.state=self.valid(self.b,self.state,np.array([[-1,-1]]),self.turn)
                    if self.turn==1:
                        print('Black must pass.')
                    else:
                        print('White must pass.')
                    self.turn = 3-self.turn
                    self.move +=1
                    if self.number_of_valid_moves(self.b,self.state,self.turn)==0:
                        self.game_in_progress=0
                        self.display_result()
                else:
                    self.game_in_progress=0
                    self.display_result()
        if self.game_in_progress:
            w,end_game,_,_=self.winner(self.b,self.state)
            if end_game==1:
                self.game_in_progress=0
                self.display_result()
        plt.draw()

    def display_result(self):
        w,_,_,_=self.winner(self.b,self.state)
        if self.human_players == 1:
            if w==3-self.my_color:
                tkMessageBox.showwarning('Message', 'I won.')
            elif w==self.my_color:
                tkMessageBox.showwarning('Message', 'You won.')
            else:
                tkMessageBox.showwarning('Message', 'Tie.')
        else:
            if w==1:
                tkMessageBox.showwarning('Message', 'Black won.')
            elif w==2:
                tkMessageBox.showwarning('Message', 'White won.')
            else:
                tkMessageBox.showwarning('Message', 'Tie.')

    def draw_board(self):
        for xx in range(self.nx):
            for yy in range(self.ny):
                if self.b[xx, yy, 0] == 1:
                    self.stone[xx, yy].set_facecolor('black')
                    self.stone[xx, yy].set_visible(True)
                elif self.b[xx, yy, 0] == 2:
                    self.stone[xx, yy].set_facecolor('white')
                    self.stone[xx, yy].set_visible(True)
                else:
                    self.stone[xx, yy].set_visible(False)

    def draw_newboard(self, ax):
        nx = self.nx
        ny = self.ny

        if self.board_type == 'go':
            rect = patches.Rectangle((0.5, 0.5), float(nx), float(ny), linewidth = 1,\
                    edgecolor = [255.0/255, 239.0/255, 173.0/255],\
                    facecolor = [255.0/255, 239.0/255, 173.0/255])
            ax.add_patch(rect)

            for kk in range(nx):
                plt.plot([kk + 1.0, kk + 1.0], [1, ny], color = [0,0,0])

            for kk in range(ny):
                plt.plot([1, nx], [kk + 1.0, kk + 1.0], color = [0,0,0])
        elif self.board_type == 'basic':
            ax.add_patch(patches.Rectangle((0.5, 0.5), float(nx), float(ny), linewidth = 1,\
                        edgecolor = [0, 0, 0], facecolor = self.board_color))

            for kk in range(nx+1):
                plt.plot([float(kk) + 0.5, float(kk) + 0.5], [0.5, ny + 0.5], color = 'black')

            for kk in range(ny+1):
                plt.plot([0.5, nx + 0.5], [kk + 0.5, kk + 0.5], color = 'black')
        else:
            for xx in range(nx):
                for yy in range(ny):
                    if np.mod(xx+yy+1, 2) == 0:
                        ax.add_patch(patches.Rectangle((float(xx) + 0.5,\
                                float(yy) + 0.5), 1.0, 1.0,\
                                facecolor = [102.0/255, 68.0/255, 46.0/255],\
                                edgecolor = 'black', linewidth=1))
                    else:
                        ax.add_patch(patches.Rectangle((float(xx) + 0.5,\
                            float(yy) + 0.5), 1.0, 1.0,\
                            facecolor = [247.0/255,236.0/255,202.0/255],\
                            edgecolor = 'black',linewidth=1))

        for xx in range(nx):
            for yy in range(ny):
                self.stone[xx, yy] = patches.Circle((xx + 1.0, yy + 1.0), 0.4,\
                        facecolor = 'black', edgecolor = 'black',\
                        visible = False, zorder = 10)
                self.txt[xx,yy] = plt.text(xx + 1.0, yy + 1.0, '',\
                        horizontalalignment = 'center',\
                        verticalalignment = 'center',\
                        fontsize = 15,\
                        color = 'blue', visible = False, zorder = 20)
                ax.add_patch(self.stone[xx, yy])

        self.txt_winner = plt.text((nx + 1.0) / 2, (ny + 1.0) / 2, '',\
                horizontalalignment='center', fontsize = 30,\
                color = 'blue', visible = False)

    def init_board(self):
        for xx in range(self.nx):
            for yy in range(self.ny):
                self.stone[xx, yy].set_visible(False)
                self.txt[xx, yy].set_visible(False)
        self.txt_winner.set_visible(False)

    def show_probabilities(self, wp_all, ox, oy):
        if self.show_p:
            for xx in range(self.nx):
                for yy in range(self.ny):
                    if xx == ox and yy == oy:
                        self.txt[xx, yy].set_visible(False)

                    if wp_all[xx, yy] >= 0:
                        self.txt[xx, yy].set_text(round(wp_all[xx, yy] * 100.0))
                        self.txt[xx, yy].set_visible(True)
                    else:
                        self.txt[xx, yy].set_visible(False)

    def hide_probabilities(self):
        for xx in range(self.nx):
            for yy in range(self.ny):
                self.txt[xx, yy].set_visible(False)

    def number_of_valid_moves(self, b, state, p):
        nx, ny, ng = np.shape(b)
        nv = np.zeros((ng, 1))
        for x in range(nx):
            for y in range(ny):
                r, _, _ = self.valid(b, state, self.xy(nx * x + y), p)
                nv += r
        return nv

    def xy(self, k): # xy position
        if hasattr(k, '__len__'):
            n = len(k)
        else:
            n = 1
        ixy = np.zeros((n, 2))
        ixy[:, 0] = np.floor(k / float(self.ny))
        ixy[:, 1] = np.mod(k, self.ny)
        return ixy

class game1(board_game):
    def __init__(self, nx = 5, ny = 5, name = 'simple go'):
        self.nx = nx
        self.ny = ny
        self.name = name

    def game_init(self, ng):
        # Initialize board for simple go game
        # Inputs
        #   nx, ny: board size
        #   ng: number of boards
        # Return values
        #   b: board
        #   state: state for b

        b = np.zeros([self.nx,self.ny,ng])
        state = -np.ones([2,ng])
        return b, state
    def winner(self, b, state):
        # Inputs
        #   b: current board state, 0: no stone, 1: black, 2: white
        #   state: extra state
        # Return values
        #   [r, end_game, s1, s2] = winner(b, state)
        #   r
        #       0: tie
        #       1: black wins
        #       2: white wins
        #       This is the current winner.
        #       This may not be the final winner.
        #   end_game
        #       1 : game ends
        #       0 : game ends if no more move is possible for the current player.
        #       -1: game ends if no move is possible for both players.
        #   for game1, 'end_game' will be always zero
        #   s1 
        #       score for black
        #   s2 
        #       score for white
        ng = b.shape[2]
        nx = self.nx; ny = self.ny
        r = np.zeros((ng))
        s1 = np.zeros((ng))
        s2 = np.zeros((ng))
        f = [[0,1,0],[1,1,1],[0,1,0]]
        b_temp = np.zeros([nx,ny])
        for j in range(ng):
            b_temp = np.array(b[:,:,j])
            e = np.ones([nx+2, ny+2])
            e[1:nx+1,1:ny+1] = 1.0 * (b_temp==1)
            g = conv2d(e, f, mode = 'valid')
            s1[j] = np.sum((g==4) * (b_temp==0))
            e[1:nx+1,1:ny+1] = 1.0 * (b_temp==2)
            g = conv2d(e, f, mode = 'valid')
            s2[j] = np.sum((g==4) * (b_temp==0))
        r = (s1>s2) + 2 * (s2>s1)
        return r, -np.ones((ng)), s1, s2

    def valid(self, b, state, xy, p):
        # Check if the move (x,y) is valid.
        # Inputs
        #   b: current board state, 0: no stone, 1: black, 2: white
        #   state: extra state
        #   xy=[xs, ys]: new position
        #   p: current plyaer, 1 or 2
        # Return values
        #   [r,new_board,new_state] = valid(b,state,[xs,ys],p)
        #   r
        #       1: valid
        #       0: invalid
        #   new_board: update board state
        #   new_state: update extra state
        ng = b.shape[2]
        nx = self.nx; ny = self.ny

        if len(xy) < ng:
            xs = np.ones((ng)) * xy[:,0]
            ys = np.ones((ng)) * xy[:,1]
        else:
            xs = xy[:,0]
            ys = xy[:,1]

        # whether position is valid or not
        r = np.zeros((ng))
        new_board = np.zeros(b.shape)
        new_board[:,:,:] = b[:,:,:] # copy by values
        new_state = -np.ones([2,ng])

        o = 3-p #opponent
        sx = nx + 2
        sy = ny + 2

        for j in range(ng):
            x = int(xs[j])
            y = int(ys[j])
            
            if x == -1 or y == -1:
                continue
            if x == state[0,j] and y == state[1,j]: # prohibited due to pai; ko
                continue
            b1 = np.zeros([nx,ny]);
            b1[:,:] = b[:,:,j]
            b2 = np.zeros(b1.shape)
            if b1[x,y] == 0:
                r[j] = 1;
                b1[x,y] = p
                [opponent_captured, xc, yc, b1] = self.check_captured4(x,y,o,b1)
                if opponent_captured == 0:
                    if self.check_captured(x,y,b1): # if suicide move?
                        r[j] = 0 # invalid
                        b1[x,y] = 0
                    b1 = (b1 * (b1>0)) + (p * (b1<0))
                    if r[j]:
                        e = np.zeros([nx+2,ny+2]) + p
                        e[1:nx+1,1:ny+1] = b1
                        # reducing own single-size territory is not allowed unless
                        # it prevents the opponent from playing there to capture stones
                        if e[x,y+1] == p and e[x+2,y+1] == p and e[x+1,y] == p\
                                and e[x+1,y+2] == p:
                            b2[:,:] = b1[:,:]
                            b1[x,y] = o
                            [n_captured_temp, xc_temp, yc_temp, b1] =\
                                self.check_captured4(x,y,p,b1)
                            b1[:,:] = b2[:,:]
                            if n_captured_temp > 0:
                                r[j] = 1
                                b1[x,y] = p
                            else:
                                r[j] = 0
                                b1[x,y] = 0
                elif opponent_captured == 1:
                    b2[:,:] = b1[:,:]
                    b1[xc,yc] = o
                    [n_captured_temp, xc_temp, yc_temp, b1] =\
                            self.check_captured4(xc, yc, p, b1)
                    b1[:,:] = b2[:,:]
                    if n_captured_temp == 1:
                        new_state[0,j] = xc
                        new_state[1,j] = yc
                new_board[:,:,j] = b1[:,:]
        return r, new_board, new_state

    def check_captured4(self, x, y, c, b1):
        nx = self.nx; ny = self.ny
        xc = x
        yc = y
        n_captured = 0
        if x > 0 and b1[x-1,y] == c:
            if self.check_captured(x-1, y, b1):
                n_captured = n_captured + np.sum(b1 < 0)
                xc = x - 1
                b1 = b1 * (b1>0) + 0
            else:
                # revert
                b1 = (b1 * (b1>0)) + (c * (b1<0))
        if y > 0 and b1[x,y-1] == c:
            if self.check_captured(x, y-1, b1):
                n_captured = n_captured + np.sum(b1 < 0)
                yc = y - 1
                b1 = b1 * (b1>0) + 0
            else:
                # revert
                b1 = (b1 * (b1>0)) + (c * (b1<0))
        if x < nx-1 and b1[x+1,y] == c:
            if self.check_captured(x+1, y, b1):
                n_captured = n_captured + np.sum(b1 < 0)
                xc = x + 1
                b1 = b1 * (b1>0) + 0
            else:
                # revert
                b1 = (b1 * (b1>0)) + (c * (b1<0))
        if y < ny-1 and b1[x,y+1] == c:
            if self.check_captured(x, y+1, b1):
                n_captured = n_captured + np.sum(b1 < 0)
                yc = y + 1
                b1 = b1 * (b1>0) + 0
            else:
                # revert
                b1 = (b1 * (b1>0)) + (c * (b1<0))
        return n_captured, xc, yc, b1

    def check_captured(self, x, y, b1):
        nx = self.nx; ny = self.ny
        captured = 1
        c = b1[x,y]
        b1[x,y] = -1
        if x > 0 and b1[x-1,y] == 0:
            captured = 0
            b1[x,y] = c
            return captured 
        if y > 0 and b1[x,y-1] == 0:
            captured = 0
            b1[x,y] = c
            return captured
        if x < nx-1 and b1[x+1,y] == 0:
            captured = 0
            b1[x,y] = c
            return captured
        if y < ny-1 and b1[x,y+1] == 0:
            captured = 0
            b1[x,y] = c
            return captured
        if x > 0:
            if b1[x-1,y] == c:
                captured = self.check_captured(x-1, y, b1)
            if not captured:
                b1[x,y] = c
                return captured
        if y > 0:
            if b1[x,y-1] == c:
                captured = self.check_captured(x, y-1, b1)
            if not captured:
                b1[x,y] = c
                return captured
        if x < nx-1:
            if b1[x+1,y] == c:
                captured = self.check_captured(x+1, y, b1)
            if not captured:
                b1[x,y] = c
                return captured
        if y < ny-1:
            if b1[x,y+1] == c:
                captured = self.check_captured(x, y+1, b1)
            if not captured:
                b1[x,y] = c
                return captured
        return captured


class game2(board_game):
    def __init__(self, nx = 3, ny = 3, n = 3, name = 'tic tac toe', theme = 'basic'):
        self.nx = nx
        self.ny = ny
        self.n = n # n-mok
        self.name = name
        self.theme = theme

    def winner(self, b, state):
        # Check who wins for n-mok game
        # Inputs
        #    b: current board state, 0: no stone, 1: black, 2: white
        #    state: extra state
        # Usage) [r, end_game, s1, s2]=winner(b, state)
        #    r: 0 tie, 1: black wins, 2: white wins
        #    end_game
        #        if end_game==1, game ends
        #        if end_game==0, game ends if no more move is possible for the current player
        #        if end_game==-1, game ends if no moves are possible for both players
        #    s1: score for black
        #    s2: score for white

        # total number of games
        ng = b.shape[2]
        n = self.n
        r = np.zeros((ng))
        fh = np.ones((n, 1))
        fv = np.transpose(fh)
        fl = np.identity(n)
        fr = np.fliplr(fl)
        for j in range(ng):
            c = (b[:, :, j] == 1)
            if np.amax(conv2d(c, fh, mode = 'valid') == n)\
                or np.amax(conv2d(c, fv, mode = 'valid') == n)\
                or np.amax(conv2d(c, fl, mode = 'valid') == n)\
                or np.amax(conv2d(c, fr, mode = 'valid') == n):
                r[j] = 1

            c = (b[:, :, j] == 2)
            if np.amax(conv2d(c, fh, mode = 'valid') == n)\
                or np.amax(conv2d(c, fv, mode = 'valid') == n)\
                or np.amax(conv2d(c, fl, mode = 'valid') == n)\
                or np.amax(conv2d(c, fr, mode = 'valid') == n):
                r[j] = 2
        return r, r > 0, r == 1, r == 2

    def valid(self, b, state, xy, p):
        # Check if the move (x,y) is valid for a basic game where any empty board position is possible.
        # Inputs
        #    b: current board state, 0: no stone, 1: black, 2: white
        #    state: extra state
        #    xy=[xs, ys]: new position
        #    p: current player, 1 or 2
        # Return values
        #    [r,new_board,new_state]=valid(b,state,(xs,ys),p)
        #    r: 1 means valid, 0 means invalid
        #    new_board: updated board state
        #    new_state: updated extra state
        ng = b.shape[2]
        n = self.n
        if len(xy) < ng:
            xs = np.ones((ng)) * xy[:, 0]
            ys = np.ones((ng)) * xy[:, 1]
        else:
            xs = xy[:, 0]
            ys = xy[:, 1]

        # whether position is valid or not
        r = np.zeros((ng))
        new_board = np.zeros(b.shape)
        new_board[:, :, :] = b[:, :, :] # copy by values
        for j in range(ng):
            x = int(xs[j])
            y = int(ys[j])

            if x == -1 or y == -1:
                continue
            if b[x, y, j] == 0: # position is empty in the j-th game
                r[j] = 1
                new_board[x, y, j] = p 

        return r, new_board, state

class game3(board_game):
    def __init__(self, nx = 9, ny = 9, n = 5, name = '5-mok'):
        self.nx = nx
        self.ny = ny
        self.n = n # n-mok
        self.name = name

    def winner(self, b, state):
        # Check who wins for n-mok game
        # Inputs
        #    b: current board state, 0: no stone, 1: black, 2: white
        #    state: extra state
        # Usage) [r, end_game, s1, s2]=winner(b, state)
        #    r: 0 tie, 1: black wins, 2: white wins
        #    end_game
        #        if end_game==1, game ends
        #        if end_game==0, game ends if no more move is possible for the current player
        #        if end_game==-1, game ends if no moves are possible for both players
        #    s1: score for black
        #    s2: score for white

        # total number of games
        ng = b.shape[2]
        n = self.n
        r = np.zeros((ng))
        fh = np.ones((n, 1))
        fv = np.transpose(fh)
        fl = np.identity(n)
        fr = np.fliplr(fl)

        for j in range(ng):
            c = (b[:, :, j] == 1)
            if np.amax(conv2d(c, fh, mode = 'valid') == n)\
                    or np.amax(conv2d(c, fv, mode = 'valid') == n)\
                    or np.amax(conv2d(c, fl, mode = 'valid') == n)\
                    or np.amax(conv2d(c, fr, mode = 'valid') == n):
                r[j] = 1

            c = (b[:, :, j] == 2)
            if np.amax(conv2d(c, fh, mode = 'valid') == n)\
                    or np.amax(conv2d(c, fv, mode = 'valid') == n)\
                    or np.amax(conv2d(c, fl, mode = 'valid') == n)\
                    or np.amax(conv2d(c, fr, mode = 'valid') == n):
                r[j] = 2

        return r, r > 0, r == 1, r == 2

    def valid(self, b, state, xy, p):
        # Check if the move (x,y) is valid for a basic game where any empty board position is possible.
        # Inputs
        #    b: current board state, 0: no stone, 1: black, 2: white
        #    state: extra state
        #    xy=[xs, ys]: new position
        #    p: current player, 1 or 2
        # Return values
        #    [r,new_board,new_state]=valid(b,state,(xs,ys),p)
        #    r: 1 means valid, 0 means invalid
        #    new_board: updated board state
        #    new_state: updated extra state
        ng = b.shape[2]
        n = self.n

        if len(xy) < ng:
            xs = np.ones((ng)) * xy[:, 0]
            ys = np.ones((ng)) * xy[:, 1]
        else:
            xs = xy[:, 0]
            ys = xy[:, 1]

        # whether position is valid or not
        r = np.zeros((ng))
        new_board = np.zeros(b.shape)
        new_board[:, :, :] = b # copy by values
        for j in range(ng):
            x = int(xs[j])
            y = int(ys[j])

            if x == -1 or y == -1:
                continue
            if b[x, y, j] == 0: # position is empty in the j-th game
                r[j] = 1 
                new_board[x, y, j] = p 

        return r, new_board, state

class game4(board_game):
    def __init__(self, nx = 8, ny = 8, name = 'othello', theme = 'basic'):
        # Initialize the class "game4".
        # Inputs
        #   nx, ny: board size
        self.nx = nx
        self.ny = ny
        self.name = name
        self.theme = theme

    def game_init(self, ng):
        # Initialize board for Othello
        # Inputs
        #   nx, ny: board size
        #   ng: number of boards
        # Return values
        #   b: board
        #   state: extra state
        if self.nx < 3 or self.ny < 3:
            raise Exception('Board is too small')

        b = np.zeros((self.nx, self.ny, ng))
        sx = int(np.floor(self.nx / 2) - 1)
        sy = int(np.floor(self.ny / 2) - 1)
        b[sx, sy, :] = 1
        b[sx + 1, sy, :] = 2
        b[sx, sy + 1, :] = 2
        b[sx + 1, sy + 1, :] = 1
        state = np.zeros((0, ng))

        return b, state

    def winner(self, b, state):
        # Check who wins for Othello
        # Inputs
        #   b: current board state, 0: no stone, 1: black, 2: white
        #   state: extra state
        # Return values
        #   [r, end_game, s1, s2] = winner(b, state)
        #   r
        #       0: tie,
        #       1: black wins
        #       2: white wins
        #       This is the current winner.
        #       This may not be the final winner.
        #   end_game
        #       1 : game ends
        #       0 : game ends if no more move is possible for the current player.
        #       -1: game ends if no move is possible for both players. 
        #   s1
        #       score for black
        #   s2
        #       score for white
        ng = b.shape[2]
        r = np.zeros((ng))
        end_game = np.zeros((ng, 1))

        s0 = np.squeeze((b == 0).sum(axis = 0, keepdims = True).sum(axis = 1,\
                keepdims = True))

        s1 = np.squeeze((b == 1).sum(axis = 0, keepdims = True).sum(axis = 1,\
                keepdims = True))

        s2 = np.squeeze((b == 2).sum(axis = 0, keepdims = True).sum(axis = 1,\
                keepdims = True))

        r = (s1 > s2) + (s2 > s1) * 2

        return r, 1 - 2 * (s0 > 0) * (s1 > 0) * (s2 > 0), \
                s1, s2
        # For the second argument,
        # returns 1, if the board is full or one of the players have no stones,
        # retures -1, otherwise.

    def valid(self, b, state, xy, p):
        # Check if the move (x,y) is valid for Othello
        # Inputs
        #   b: current board state
        #       0: no stone
        #       1: black
        #       2: white
        #   state: extra state
        #   xy = [xs, ys]: new position
        #   p: current player, 1 or 2
        # Return values
        #   [r, new_board, new_state] = valid(b, state, xy, p)
        #   r
        #       1: valid
        #       0: invalid
        #   new_board: updated board state
        #   new_state: updated extra state
        ng = b.shape[2]

        if len(xy) < ng:
            xs = np.ones((ng)) * xy[:, 0]
            ys = np.ones((ng)) * xy[:, 1]
        else:
            xs = xy[:, 0]
            ys = xy[:, 1]

        # whether position is valid or not
        r = np.zeros((ng))
        new_board = np.zeros(b.shape)
        b1 = np.zeros((b.shape[0], b.shape[1]))

        new_board[:, :, :] = b[:, :, :]

        dx = np.array([1,  1,  0, -1, -1, -1,  0,  1])
        dy = np.array([0, -1, -1, -1,  0,  1,  1,  1])

        for j in range(ng):
            x = int(xs[j])
            y = int(ys[j])

            if x == -1 or y == -1:
                continue
            if b[x, y, j] == 0:
                b1[:, :] = b[:, :, j]
                v = 0
                for z in range(len(dx)):
                    if self.check_captured(x, y, dx[z], dy[z], p, b1):
                        v = 1
                        b1 = b1 * (b1 > 0) + p * (b1 < 0)
                    else:
                        b1 = b1 * (b1 > 0) + (3 - p) * (b1 < 0) # restore
                if v:
                    r[j] = 1
                    b1[x, y] = p
                    new_board[:, :, j] = b1[:, :]

        return r, new_board, state

    def check_captured(self, x, y, dx, dy, c, b1):
        r = 0
        o = 0
        for a in range(min(self.nx, self.ny)):
            x = x + dx
            y = y + dy
            if x >= 0 and x < self.nx and y>= 0 and y < self.ny:
                if b1[x, y] == c:
                    if o:
                        r = 1
                    return r
                elif b1[x, y] == 3 - c:
                    o = 1
                    b1[x, y] = -1
                else:
                    return r
            else:
                return r
