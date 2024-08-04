<h1> Dice </h1>
C++ class Dice(dice.cpp) is wrapped and can be used with python via "import dice" header. C++ Dice objects transformed into Python pyDice objects. <br>
Wrapping is done through cython.
---
<h2> Sample Run </h2>

After dependencies are satisfied, cd into the folder (dice). <br>
<!--style:CustomCodeFence-->
~~~
cd ./dice
~~~

Then, then these commands can  be used to run the sample code "testDice.py" which includes several class methods.

```
python setup.py build_ext --inplace

python testDice.py 
```
This is the result of a sample run 
~~~
5
2
Result of dual roll: 9
Player 2 wins with 6
Player 1 loses with 2
~~~
