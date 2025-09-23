# 4CE

// Under Development //

## Superhuman AI to beat the game 4CE

The aim of this project is to train an AI agent through self-play to achieve superhuman performance on the game of 4CE.
The setting is similar to the one in (Silver et al., 2017, https://www.nature.com/articles/nature24270), since we use no human data to empower the algorithm. Note however, that this repository is not a reimplementation of the above, rather an exploration of self-play using approaches based on intuition and curiosity. The approach currently relies on DQN as a key algorithm (Mnih et al., 2013, https://arxiv.org/abs/1312.5602).

### 4CE

4CE is derived from the classic game of Tic-Tac-Toe by extending the board to 4 dimensions and letting players play as long as there is any empty square left on the board. Players take alternating moves, and whoever manages to form a row of three from their symbol along any dimension scores a point.
For fairness, the center square is considered unplayable. Therefore the board consists of $3^4 - 1 = 80$ squares.
The player with the most points at the end of the game wins.
