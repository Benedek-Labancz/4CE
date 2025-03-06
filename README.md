# 4CE
An extended API for the game 4CE.
Under development.


Example usage:
```
import os
from game import Game, print_state, print_score

game = Game()

done = False
while not done:
    
    os.system("cls||clear")
    print_state(game.state)
    print_score(game.score)

    action = input().split(' ') # input action in form "K L x y"
    action = tuple(int(x) for x in action)
    new_state, reward, done, info = game.step(action)

    game.state = new_state
    game.switch_turn()
```