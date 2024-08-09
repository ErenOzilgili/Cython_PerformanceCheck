import dice as dc

dice1 = dc.pyDice()
dice2 = dc.pyDice(6)

score1 = dice1.Roll()
score2 = dice2.Roll()

print(dice1.NumSides())
print(score1)

print("Result of dual roll: {0}".format(dice1.DualRoll(dice2)))

if score1 > score2:
    print("Player 1 wins with {0}".format(score1))
    print("Player 2 loses with {0}".format(score2))
elif score1 == score2:
    print("It's a tie with score of {0}".format(score1))
else:
    print("Player 2 wins with {0}".format(score2))
    print("Player 1 loses with {0}".format(score1))
