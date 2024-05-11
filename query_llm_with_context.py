import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

---

Leisure Genius presents

THE COMPUTER EDITION OF MONOPOLY(r)

Parker Brothers Real Estate Trading Game

For the Commodore 64 & 128
(r) Registered Trade Mark


MONOPOLY(r)


INTRODUCTION and RULES                 PAGE  1
RULES for a SHORT GAME                 PAGE  8
INTRODUCTION TO COMPUTER MONOPOLY(r)   PAGE  9
LOADING INSTRUCTIONS                   PAGE  9
STARTING A GAME                        PAGE 10
SCREEN DESCRIPTION                     PAGE 11
INDEX                                  PAGE 22


THE DISTINCTIVE DESIGN OF THE GAME BOARD, AS WELL AS EACH OF THE
DISTINCTIVE ELEMENTS OF THE BOARD, THE PIECES, AND THE WORD MONOPOLY(r)
ARE TRADEMARKS OF PARKER BROTHERS, DIVISION OF KENNER TOYS INC., FOR ITS
REAL ESTATE TRADING GAME AND GAME EQUIPMENT.

(c) 1935, 1985 PARKER BROTHERS, DIVISION OF KENNER PARKER TOYS INC.,
(KPT), BEVERLY, MA 01915

Leisure Genius is a wholly owned subsidiary of Virgin Mastertronic
International. (c) 1989 Leisure Genius. Published by Virgin Mastertronic
International. All rights reserved. No part of this program, packaging or
documentation may be reproduced in any form. Unauthorised copying,
hiring, lending or sale and repurchase prohibited.


MONOPOLY(r)

(r) Registered Trade Mark

Parker Brothers MONOPOLY(r) Real Estate Trading Game was presented to
Parker Brothers during the Great Depression by Charles B. Darrow of
Germantown, Pennsylvania. The Street names on the game board were adopted
from actual streets in Atlantic City, New Jersey, Darrow's favourite
vacation spot.

The first games were handmade. Darrow gave them to friends and sold a few
through a Philadelphia department store. As demand for the game grew,
Darrow could not keep up with the orders and arranged for Parker Brothers
to take over the game.

Since 1935 when Parker Brothers acquired the rights to the game, it has
become the leading proprietary game not only in the United States but
throughout the Western world. The game is published under license in
twenty-nine countries and in twenty-two foreign languages.


OBJECT... The object of the game is to become the wealthiest player
through buying, renting and selling property.


EQUIPMENT... The equipment consists of the on-screen board, two dice,
eight tokens or playing pieces, thirty-two green houses and twelve red
hotels, Chance and Community Chest cards and Title Deeds for each
property.


PREPARATION... Each player chooses one token to represent him on his
travels around the board. Each player receives $1,500 from the Bank. All
other equipment goes to the Bank.


BANKER... The computer will always act as Banker.


THE BANK... Besides the Bank's money, the Bank holds the Title Deed cards
and houses and hotels prior to purchase and use by the players. The Bank
pays salaries and bonuses. It sells and auctions properties to the
players and distributes their proper Title Deed cards; it sells houses
and hotels to the players, loans money when required on mortgages.

The Bank collects all taxes, fines, loans and interest and will buy back
houses and hotels at half price.


THE PLAY... Each player in turn throws the dice. The player with the
highest total starts the play. His token is placed on the corner marked
GO. He throws the dice and moves his token in the direction of the arrow
the number of spaces indicated by the dice. After he has completed his
play, the turn to play goes in the order in which the names have been
entered. The tokens remain on the spaces occupied and proceed from that
point on the player's next turn. One or more tokens may rest on the same
space at the same time.

Depending on the space which his token reaches, a player may be entitled
to buy real estate or other properties - or be obliged to pay rent, pay
taxes, draw a Chance or Community Chest card, "GO TO JAIL", etc.

If a player throws doubles, his token is moved the sum of the two dice
and is subject to any privileges or penalties pertaining to the space on
which he lands. He then throws again and moves his token as before. If a
player throws doubles three times in succession, he must "GO TO JAIL".


GO... Each time that a player's token lands on or passes over "GO", the
Bank pays him $200 "Salary".


BUYING PROPERTY... Whenever a player lands on an unowned property he may
buy that property from the Bank at its displayed price. He receives the
Title Deed card showing ownership. If he does not wish to buy the
property it is sold at auction by the Banker to the highest bidder. The
buyers pay to the Bank the amount of the bid in cash and receives the
Title Deed card for that property. Any player, including the one who
declined the option of buying it at the printed price, may bid. Bidding
may start at any price.


PAYING RENT... When a player lands on property owned by another player
the owner collects rent from him in accordance with the list printed on
the Title Deed card applying to it.

If the property is mortgaged, no rent can be collected.

It is an advantage to hold all the Title Deeds in a color group because
the owner may then charge double rent for unimproved properties in that
color group. (ie. Boardwalk and Park Place, or Connecticut, Vermont and
Oriental Avenues). This rule applies to unmortgaged properties even if
another property in that color group is mortgaged.

It is even more of an advantage to have houses or hotels on properties
because rents are much higher than for unimproved properties.

The owner may not collect his rent if he fails to ask for it before the
next player throws the dice.


CHANGE and COMMUNITY CHEST... When a player lands on either of these
spaces he is allocated the next card from the pack and follows the
instructions.

The "Get Out of Jail Free" card is held until needed. After being used it
is "returned" to the pack.


INCOME TAX... When a player lands on "Income Tax" he has two options; he
may estimate his tax at $200 and pay the Bank, or he may pay 10% of his
total worth to the Bank. His total worth is all his cash on hand, printed
prices of mortgaged and unmortgaged properties and cost price of all
buildings he owns.

The player must decide which option he will take before he adds up his
total worth.


JAIL... A player lands in Jail when...
(1) his token lands on the space marked "GO TO JAIL"
(2) he is allocated a card marked "GO TO JAIL"
(3) he throws doubles three times in succession.

When a player is sent to Jail he cannot collect $200 salary in that move
since regardless of where his token is on the board, he must move it
directly into Jail. A player's turn ends when he is sent to Jail.

If a player is not sent to Jail but in the ordinary course of play lands
on that space, he is "Just Visiting", incurs no penalty, and moves ahead
in the usual manner on his next turn.

A player gets out of Jail by...
(1) throwing doubles on any of his next three turns. (If he succeeds in
    doing this he immediately moves forward the number of spaces shown by
    his doubles throw. Even through he has thrown doubles he does not
    take another turn.)
(2) using the "Get Out of Jail Free" card if he has it.
(3) purchasing the "Get Out of Jail" card from another player and playing
    it.
(4) paying a fine of $50 before he throws the dice of either of his next
    two turns.

If the player does not throw doubles by his third turn he must pay the
$50 fine. He then gets out of Jail and immediately moves forward the
number of spaces shown by his throw.

Even though he is in Jail, a player may buy or sell property, buy or sell
houses and collect rents.


FREE PARKING... A player landing on this space does not receive any
money, property or reward of any kind. This is just a "free" resting
place.


HOUSES... When a player owns all the properties in a color group he may
buy houses from the Bank and erect them on those properties.

If he buys one house, he may put it on any one property. The next house
he buys must be erected on one of the unimproved properties of this or
any other complete color group he may own.

The price he must pay the Bank for each house is shown on his Title Deed
card for the property on which he erects the house.

The owner can still collect double rent from an opponent who lands on the
unimproved properties of his complete color group.

A player may buy and erect at any time as many houses as his judgement
and finances will allow. But he must build evenly (i.e. he cannot erect
more than one house on any one property of any color group until he has
built one house on every property of that group. He may then begin on the
second row of houses, and so on, up to a limit of four houses to a
property. For example, he cannot build three houses on one property if he
has only one house on another property of that group).

As a player builds evenly, he must also break down evenly if he sells
houses back to the Bank (see SELLING PROPERTY).


HOTELS... When a player has four houses on each property of a complete
color group, he may buy a hotel from the Bank and erect it on any
property of that color group. He returns the four houses from that
property to the Bank and pays the price for the hotel as shown on the
Title Deed card. Only one hotel may be erected on any one property.


BUILDING SHORTAGE... When the Bank has no houses to sell, players wishing
to build must wait for some player to turn back or to sell his houses to
the Bank before building. If there are a limited number of houses and
hotels available, and two or more players wish to buy more than the Bank
has, the houses or hotels must be sold by auction to the highest bidder.


SELLING PROPERTY... Unimproved properties, railroads and utilities (but
not buildings) may be sold to any player as a private transaction for any
amount that the owner can get. However, no property can be sold to
another player if buildings are standing on any properties of that color
group. Any buildings so located must be sold back to the Bank before the
owner can sell any property of that color group.

Houses and hotels may be sold back to the Bank at any time for one half
the price paid for them.

All houses on one color group may be sold one by one, evenly, in the
reverse order in which they were erected. All hotels on one color group
may be sold at once. Or they may be sold evenly, in reverse order to
which they were erected.


MORTGAGES... Unimproved properties can be mortgaged through the Bank at
any time. Before an improved property can be mortgaged all the buildings
on all the properties of its color group must be sold back to the Bank at
half price. The mortgage value is displayed on each Title Deed card. No
rent can be collected on mortgaged properties or utilities, but rent can
be collected on unmortgaged properties in the same group.

In order to lift the mortgage, the owner must pay the Bank the amount of
the mortgage plus 10% interest. When all the properties of a color group
are no longer mortgaged the owner may begin to buy back houses at full
price.

The player who mortgages property retains possession of it and no other
player may secure it by lifting the mortgage from the Bank. However, the
owner may sell this mortgaged property to another player at any agreed
price. The new owner may lift the mortgage at once, if he wishes, by
paying off the mortgage plus 10% interest to the Bank. If he does not
lift the mortgage at once he must pay the Bank 10% interest when he buys
the property and if he lifts the mortgage later he must pay an additional
10% interest as well as the amount of the mortgage to the Bank.


BANKRUPTCY... A player is bankrupt when he owes more than he can pay
either to another player or to the Bank.

If his debt is to another player, he must turn over to that player all
that he has of value and retire from the game. In the making of this
settlement, if he owns houses or hotels, he must return these to the Bank
in exchange for one half the amount of money paid for them. This cash is
given to the creditor.

If he has mortgaged property he also turns this property over to his
creditor but the new owner must at once pay the Bank the amount of
interest on the loan, which is 10% of the value of the property.

After the new owner does this, he may, at his option, pay the principal
or hold the property until some later turn at which time he may lift the
mortgage.

Should a player owe the Bank, instead of another player, more than he can
pay (because of taxes or penalties) even by selling his buildings and
mortgaging property, he must turn over all his assets to the Bank. The
Bank immediately sells by auction all property so taken, except buildings.

A bankrupt player must immediately retire from the game.

The last player left in the game wins.


MISCELLANEOUS... Money can only be loaned to a player by the Bank and
then only by mortgaging property.

No player may borrow from or lend to another player.


RULES FOR A SHORT GAME

Before commencing, the players agree upon a definite time to end the
game. The Banker "shuffles" and cuts the Title Deed cards and deals two
cards to each player. The players immediately pay the Bank the price of
the property dealt to them.

In this short game it is only necessary to have three houses (instead of
four) on each lot of a complete color group before the player may buy a
hotel.

Rent received for a hotel remains the same as in the regular game.

The game proceeds in the usual manner until the agreed finishing time is
reached. No further dealings must take place, but if a player is in play
when the finish is announced, he is allowed to complete his move and any
transactions in connection with it.

Each player then calculates the value of his possessions consisting of:
1) Cash in hand.
2) Unimproved property & Utilities (at purchase price).
3) Mortgaged property (at mortgage value).
4) Houses (at cost price).
5) Hotels (at the cost of five houses).

THE RICHEST PLAYER WINS!


INTRODUCTION TO COMPUTER MONOPOLY

The Computer version of the MONOPOLY board game conforms to all the rules
of the board game except that 1-8 players can participate. These may be
human or computer or any combination. The computer will also play banker
and will control all money and property transactions automatically.

Players are advised to play a test game consisting of human players only,
while they familiarize themselves with the instructions. Playing against
the computer will add a new dimension to the game as the computer
opponents play a ruthless game.

It has been necessary to abbreviate some of the names of the properties
shown on the board. The large Title Deed Cards still include the name in
full. Below is a list of the abbreviations:

MEDITERRANEAN	- MEDIT.
CONNECTICUT	- CONN.
VIRGINIA	- VIR.
PENNSYLVANIA	- PENN.
TENNESSEE	- TENN.
KENTUCKY	- KENTKY.
ILLlNOIS	- ILL.
NORTH CAROLINA	- N.CARO.
ATLANTIC	- ATLAN.
ORlENTAL	- ORIENT.


LOADING INSTRUCTIONS

If you are using a Commodore 128 then type GO 64.

Loading from disk: Place disk in drive and type LOAD "MONOPOLY",8,1 and
press RETURN.

Note: If you wish to save your game you will need to format a disk to
save it to. Refer to your Commodore User's Guide for formatting
instructions.

DO NOT SAVE GAMES TO TME MONOPOLY DISK.


STARTING A GAME

Once the game has loaded you will be prompted with the following:

1. DO YOU WISH TO LOAD A PREVIOUSLY SAVED GAME?
If you wish to resume a saved game then press Y. You will then be
prompted to enter the file name of your saved game. Enter the file name
and press RETURN. You will then be prompted to insert your SAVE game disk
and press any key.

Your game will resume from the point you saved it. If you do not wish to
load a saved game then press the N key.

2. HOW MANY PLAYERS IN THIS GAME?
You can play with 2-8 players. Any player can be controlled by a human or
the computer.

3a. PRESS 1 to 8 TO CHOOSE A TOKEN.
On the bottom of the screen you will notice the eight MONOPOLY playing
tokens - Motor Car, Thimble, etc. Press the number key corresponding to
the token you wish to play (ie. if you wish to play with the Motor Car
then press key 1, for the Thimble press key 2, etc).

3b. GIVE THIS PLAYER A NAME.
Now enter a name of not more then seven characters and press RETURN.

3c. IS THIS A COMPUTER PLAYER?
Press Y if this token is to be controlled by the computer, and N if a
human is going to play for this token.

You will now be prompted to choose another token and the whole process
begins from 3a again until all the players have been entered.

4. DO YOU WISH TO PLAY THE SHORT GAME?
The short game is played over a specified period of time. Answer Y to the
prompt and you will now be asked to ENTER THE TIME FOR A SHORT GAME IN
HOURS AND MINUTES. Enter one number for hours. Press SPACE. Then enter
two numbers for minutes. Press RETURN. The Computer will now deal two
properties to each player and deduct the money accordingly. (You are
obliged to buy the two properties dealt to you by the computer.) (See
page 14 for short game rules.)

You are now ready to start playing. The dice are now thrown in turn, and
the player throwing the highest score will go first.


SCREEN DESCRIPTION

Running along the top of the screen is the OPTION BAR. The game is
controlled from this bar. All moves start by selecting DICE and pressing
FIRE. A detailed description of all the functions on the Option Bar can
be found on page 12.

Below the Option Bar is the main playing board. This is shown in
perspective, and gives a good overall view of the game. You will notice
the player's pieces moving round each time the dice are thrown.

To the left of the main board is the name of the player whose turn it is.
The player number, and playing piece are also shown.

To the right of the main board is the clock (short game only). This
counts down and the game will terminate and winner be announced when it
reaches zero. When a player is with a transaction (Mortgage, Trade, etc)
the clock will appear to stop running. It does not update to the main
screen during transactions, however it is still running behind the scenes
and will update once the transaction has ended.

On the bottom right of the screen you will notice four cards. These give
a local view of the area in front of the player's piece, including the
square they currently occupy. These cards scroll along as the player's
piece moves around the board (only if FASTMOVE is inactive). They show
whether a property is FOR SALE, SOLD, MORTGAGED, or has HOUSES or HOTELS
built.

Above these four cards are the CASH and PROPERTY displays. CASH is
represented by a small pile of yellow coins, PROPERTY by a fan of yellow
cards. A player's cash and property are displayed during their move.

On the bottom left of the screen is an enlarged version of the square
the player currently occupies. This large card holds all details
concerning that property, Mortgage value, Building prices, and Rent
payable. If the property is owned, the owner's playing piece is displayed
in the top right corner of the card. If the property has houses or
hotels, the rent payable is highlighted in white.


CONTROL METHOD... We have tried to make this game easy to use, and have
incorporated various control methods.

The game can be played via joystick (either port), cursor keys, or
keyboard. Wherever possible we have included a default value to many of
the Y/N prompts. This enables the player to press RETURN instead of
having to decide whether to press Y or N. The defaults will become useful
to the more experienced player.

Screen messages are displayed for a period of time, if you find that this
is too long, press the SPACE-BAR. This cancels the time delay while a
message is being displayed. Don't keep pressing the SPACE-BAR or any
other keys as you will flick through all messages and prompts. The
keyboard stores all key presses in a buffer. This is a very useful
feature for the more advanced player, as they will be able to enter
certain keystrokes in advance of the prompt.

OPTION-BAR... Running across the top of the screen is the OPTION-BAR. The
game is controlled from this bar. There are three selection methods open
to the user:

1. JOYSTICK SELECTION: Move the highlight along the bar using the
   joystick, place the highlight over the option you wish to select and
   press FIRE to select it.

2. CURSOR KEY SELECTION: Move the highlight along the bar using the
   Commodore cursor keys. Press RETURN to select the highlighted option.

3. KEYBOARD SELECTION: Press the first letter of the option you wish to
   select, ie. if you wish to select the option to QUIT then press the Q
   key.

The effects of using the options are prompted within the game, but here
is a list of how to use them. Most Y/N answers have a default answer
(shown below in brackets).

Some of the options during Mortgage, Trade, and buy/selling houses can be
confusing at first due to the complexities involved. If you manage to get
stuck in a menu which is waiting for you to input a player number or sum
of money, just press RETURN and you should be put back onto the Option
Bar. This is not always the case as sometimes players are forced to
mortgage property to pay outstanding rent. A player will not be let out
of the menu until the debt has been cleared. We have tried to make the
computer game as friendly as possible with the inclusion of sensible
default answers in response to questions asked.


QUIT... This option will quit the game. Don't worry if you select it by
mistake as you are asked for verification before the QUIT command is
executed. On selecting this option you will be prompted with the following:

DO YOU WISH TO SAVE THIS GAME ? (N)
If you do not wish to save your game then press RETURN or N. See next
section (SAVE GAME) if Y.

DO YOU WISH TO CONTINUE THIS GAME ? (Y)
If you wish to continue the game then press Y or RETURN. You will now be
placed back to the Option Bar.

If you answered N, the game will end, the computer will add up all the
cash and assets of each player and declare a winner.


SAVE... This Option allows the player to SAVE the current game, and
RESUME it at a later date. You must have a previously formatted disk to
save your game onto.

DO NOT SAVE GAMES TO THE MONOPOLY DISK!!!

DO YOU WISH TO SAVE THE GAME ? (N)
If you do not wish to save your game then press N or RETURN. If you
answered Y to this then you will prompted:

ENTER FILE NAME FOR MONOPOLY DATA... Insert a blank formatted disk and
type in a file name of up 8 letters and press RETURN. You may save as
many games as you wish to disk (up to 70!) as long as you give each game
a different file name.

Note: Remember your file name as you will be asked for it when you wish
to resume a game.

DO YOU WISH TO CONTINUE THIS GAME ? (Y)
If you wish to continue the game then press Y or RETURN. You will now be
placed back to the Option Bar.


MORTGAGE... The player may either select this option voluntarily or will
be forced into it when they do not have the cash to pay an outstanding
debt. Once the option has been selected the computer asks DO YOU WANT TO
SEE THE OWNER LIST? (N) (for OWNERS option see below). After answering N
to this prompt a small flashing M (MORTGAGE) cursor will appear on the
board.

Move the cursor to the property you wish to mortgage and press RETURN. If
you wish to get out of this option press Q (QUIT). The process of
mortgaging property will continue until Q is pressed.


OWNERS... This option is very useful to keep track of what the other
players in the game are collecting. The owners list will give you either
a complete list of properties and their owners, or shortlist for an
individual player incorporating Cash, Property and record of any Jail
cards held by that player.

On entering the OWNERS option you will be prompted:

ENTER OWNER (1-8) OR (A) FOR ALL.
PRESS (Q) TO QUIT (Default Q).
If you wish to see the full owners list then press the A key.

To the left of each property in the list is the number of the player who
owns that property. A property with white lettering against a red
background denotes the property is currently mortgaged.

If you wish to view only the properties belonging to you, then you must
enter your player number at the above prompt. Your cash status is also
shown. You may check up on any other players in this way, regardless of
whether they are human or computer.


BUYING HOUSES... Select this option if you wish to buy or sell houses or
hotels. The first prompt on entering this option is:

DO YOU WISH TO BUY HOUSES ? (No Default)
If you do not wish to buy any Houses press N. If you answered Y to this
prompt you will be asked, DO YOU WISH TO SEE THE OWNER LIST?

After answering N to this question you will be prompted to ENTER A PLAYER
NO (1-8). You must enter your own number, followed by HOW MANY HOUSES DO
YOU WISH TO BUY? Any number of players may buy houses at this stage of
the game, regardless of whose turn it is. The computer will now repeat
ENTER A PLAYER NO. If another player wishes to buy Houses then they
should enter their number.

If no further players wish to buy then press RETURN.

There are 32 houses and 12 hotels in the game, if demand for houses and
hotels is greater than the Bank can supply, the Bank will hold an
auction, and the houses or hotels will be sold to the highest bidder.

After all players wishing to buy houses have entered their numbers a
flashing (HOUSE) cursor will appear on the screen. Move the cursor to the
desired color set and press FIRE. The color set will now be enlarged and
placed at the bottom right of the screen. The player now moves the cursor
over the desired property and presses fire to place a house. This
process is repeated until all houses have been placed. The computer will
automatically deduct the money to pay for the houses. If the player needs
to mortgage properties to raise the money, the computer will put them
into the mortgage option until the debt has been cleared.

The computer will let players buy houses as long as they have enough cash
and mortgagable property available to pay for them. Players who spend
over their limit of cash will be forced into the mortgage option after
the houses have been placed.


SELLING HOUSES... Selling houses follows much the same procedure as
buying. Houses are removed from the board instead of being placed.

Select the HOUSES option from the main Menu-Bar, you will be asked DO YOU
WISH TO BUY HOUSES? Answer N and you will be asked DO YOU WISH TO SELL
HOUSES? Answer Y, you are now ready to sell a house.

The first prompt inside this option is DO YOU WANT TO SEE OWNERS LIST?
You should now be familiar with how the owners list works. After
answering N to this question you will be prompted ENTER A PLAYER NO
(1-8). You must enter you own number. A flashing (HOUSE) cursor will
appear on the screen. Move the cursor to the desired color set and press
FIRE. The color set will now be enlarged and placed at the bottom right
of the screen. The player now moves the cursor over the desired property
and presses fire to remove a house. This process is repeated until the
player decides to press Q and QUIT the house selling option. The computer
will automatically pay the player half the face value of the houses sold.

Players will sometimes find they are automatically forced into the house
selling option to clear a debt.


SELLING HOTELS. . . It a player wishes to sell a Hotel the principle is
the same, unless there is a shortage of houses available from the Bank.
A player with Hotels on all properties in a Color Set may end up losing
all Hotels after selling just one.

If any Hotel is sold off and there are no Houses to replace it, the
player is forced to downgrade the Hotel to nothing. As players must build
evenly, the remaining Hotels have to be sold off to even things up. Thus
it is possible for a player to be forced to dispose of all Hotels because
of a housing shortage.

Example:
A player has a Hotel on "ST. JAMES PLACE", "TENNESSEE AVENUE", and four
Houses on "NEW YORK AVENUE". The Bank is currently holding three Houses.
(see figure 1 )

[Figure 1 omitted]

The playerg is forced to sell the Hotel on "ST. JAMES PLACE" for which
they will receive $50. The Bank only has three houses so the player
receives a further $50 tor the fourth house and has three remaining
Houses. (see figure 2)

[Figure 2 omitted]

There is now a building imbalance as "TENNESSEE AVENUE" still has a
Hotel. The Hotel on "TENNESSEE AVENUE" now has to be sold. The bank does
not have any Houses left so the whole property is effectively cleared and
the player is forced to take $250 as if the Hotel and four Houses had
been sold. The remaining Houses are now re-shuffled to create an even
balance, an odd house will always be placed on the highest valued
property by the computer. (see figure 3)

[Figure 3 omitted]

If this situation should occur during the game the Computer will
automaticaliy prompt NOT ENOUGH HOUSES LEFT TO REPLACE A HOTEL!
REARRANGING THE SET. All the neccessary calculations and re-shuffling
needed to take this rule into account will be done by the computer,
however an understanding of the rule is needed to appreciate the
strategic advantage which can be gained from forcing other players into
this situation.

Players may wish to tactically buy up all the Houses in #he hope that a
player with Hotels may need to sell one to clear a debt, the player may
end up being forced to sell all buildings.


TRADE... Players may trade with each other regardless of whether they are
controlled by the computer. Often the computer will initiate trades.
Trading is one of the most necessary parts of the game. Without it the
game can effec#ively carry on forever if all players are evenly balanced,
and no player has a color set.

The player initiating the TRADE option will be prompted:

DO YOU WANT TO SEE THE OWNER LIST ? (N)
If you are unfamiliar with this prompt, see the section on OWNERS.

After answering N to this, the computer will ask ENTER PLAYER NO. The
player wishing to trade should enter their number (1-8). If a player
wishes to get out of this option and return to the option bar, then
press RETURN instead of a number.

A flashing T (TRADE CURSOR) will now appear on the board. Use the cursor
keys or joystick to move the T cursor to the property on offer and press
RETURN.

The cursor now returns to the first property on the board. If the trading
player wishes to include more than one property in the offer, move the
cursor to the next property to trade and press RETURN. This process is
repeated until the player decides not to offer any more properties.

Each time the player presses RETURN then another property (provided it
is valid) is added to the list of properties to be traded. The trading
player may elect to offer no property in favor of paying cash only, in
which case they should press Q (QUIT). The owner of each property is
shown on the top right corner of the large card on the bottom left of
the screen. Players should only trade one propery until they have become
familiar with this option. It is possible to give property away and get
nothing in return. Computer controlled players will readily accept any
donations.

Once the offer side has been terminated by pressing the Q (QUIT) key, the
computer will prompt: ENTER PLAYER NO. This refers to the player being
traded with, (ie. if player 1 has initiated a trade with player 2, then 2
must be entered at this point.) If the player offering the trade wishes
to call off the trade then they should press RETURN.

The process of moving the flashing T cursor is now repeated as above,
only this time the player is selecting the property they wish to gain
from the trade. Again, more than one property may be included in the
exchange, also the player may wish to select no property in favor of all
cash, in which case Q (QUIT) should be pressed to terminate the property
selection.

Once the player has selected both the property on offer and property
required the computer will respond with ENTER THE CASH AMOUNT IN THIS
TRADE (RETURN if there is no cash adjustment).

If no cash adjustment is to be paid the computer will ask the player
whether they accept the offer made by the trading player. Answer Y or N.
If there is a cash adjustment then the computer will reply IS OFFER
PLAYER X TO RECEIVE THE CASH? Answer Y or N.

The computer asks the player being made the offer DO YOU ACCEPT TRADE? If
the trade is acceptable then the computer will swap the neccessary
properties/cash and the trade is completed successfully.


COMPUTER TRADES... The computer will offer only sensible trades. The
basis of the computer offering a trade is simple, it will only offer a
trade it would accept if offered by another player. If the trade is
refused it will not offer the same trade again for a number of moves.
This stops the computer persistently offering the same trades to a human
player, however the computer will constantly be looking for other trades
to offer.

Note:
Occasionally the computer will find two evenly balanced trades to offer
the same player, it may alternate between the two trades persistently.
This will only occur with the computer trading against a human who does
not wish to accept either trade, computer players will nearly always
accept offers from another computer player.

If this should happen during your game you should be patient, the
computer will time out and the next players move will begin. Next move
around the factors will have changed and the computer may well have found
a different trade.


RENT... This option is used for gathering rent owed by other players. If
a player lands on a property owned by you then it is up to you to spot it
and collect the rent. Once a player has landed on your property you are
given a small period of time to select this option, once pressed the
computer adjusts both the player's cash accordingly. If you fail to press
the RENT button then the rent will not be collected. If the other player
cannot play, the computer will make them mortgage property or sell houses
in order to clear the debt.


DICE... This option rolls the dice. You will see a graphic picture of the
rolling dice. Your piece will be moved round the board by the computer.

Depending where your playing piece lands, you will be offered a number of
choices. If you throw a double you will be allowed another throw. If you
throw three consecutive doubles then you will go to Jail.

FASTMOVE (Default Inactive)

FASTMOVE makes each player's move a little quicker. Picking this option
toggles between FASTMOVE NOW ACTIVE and FASTMOVE NOW INACTIVE.


CASH (Default Show Cash)

Picking the CASH option will toggle between showing and hiding the cash
and property value for each player at the end of their turn. This
provides an easy way of knowing the potential value of the property and
cash at a glance.

If you are playing with other human players, you may wish to switch the
visual display off. A cash and property readout may still be obtained by
going into the list of owners.

The value of property is calculated on the amount of cash that can be
raised from the Bank. An unmortgaged railroad will have a property value
of $100, the value the Bank will automatically pay if a player needs to
mortgage it.

A mortgaged railroad will effectively have a property value of $0. This
is because no more cash can be raised instantly from this property. You
may still raise cash by trading the property with another player, but
other players aren t always willing to trade.

The value of houses and hotels is also included in the PROPERTY figure,
at the value the Bank will pay for them.

NOTE: At the end of the game the final figure will not be a simple matter
of adding property and cash together. The final figure is calculated on
face value of property owned: an unmortgaged railroad at $200, mortgaged
railroad at $100, any houses and hotels at face value.

It is important that the property value shown durg the game refers to
the cash that could be raised from the property and not its face value.


AUCTIONING OF PROPERTY... If a player lands on a property which is unsold
they will be offered the option to buy it at face value. If the offer is
declined the property will be put up for auction by the computer. Players
may participate in the auction by pressing their player number (1-8) to
register a bid. Pressing the space bar will toggle between the BID and
INSTRUCTION prompts.


AUCTIONING OF HOUSES/HOTELS... There are 32 houses and 12 hotels in the
MONOPOLY game. If more than one player wishes to buy houses or hotels and
the order cannot be fulfilled due to lack of availability, the computer
will hold an auction and the remaining houses or hotels will be allocated
to the highest bidder. The auction is carried out in the same way as the
property auction.


INDEX

[Omitted, since it's useless in this format]


Limited Warranty

Virgin Mastertronic Intl. and Leisure Genius warrant to the original
purchaser of this computer software product that the recording medium on
which the software programs are recorded will be free from defects in
materials and workmanship for ninety days from the date of purchase.
Defective media returned by the purchaser during that ninety day period
will be replaced without charge, provided that the returned media have
not been subjected to misuse, damage or excessive wear.

Following the initial ninety day warranty period, defective media will
be replaced for a replacement fee of $5.00.

Defective media should be returned to: Virgin Mastertronic Intl.
                                       711 W. 17th Street, Ste. G9
                                       Costa Mesa, CA 92627

In protective packing, accompanied by (1) a brief statement describing
the defect, (2) a $5.00 check or money order (if beyond the initial
ninety day warranty period), and (3) your return address.


WHAT IS NOT COVERED BY THIS WARRANTY

THIS WARRANTY DOES NOT APPLY TO THE SOFTWARE PROGRAMS THEMSELVES. THE
PROGRAMS ARE PROVIDED "AS IS".

THIS WARRANTY IS IN LIEU OF ALL OTHER WARRANTIES, WHETHER ORAL OR
WRITTEN, EXPRESS OR IMPLIED. ANY IMPLIED WARRANTIES, INCLUDING IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
LIMITED IN DURATION TO NINETY DAYS FROM THE DATE OF PURCHASE. NEITHER
VIRGIN MASTERTRONIC INTL. NOR LEISURE GENIUS SHALL BE LIABLE FOR
INCIDENTAL OR CONSEQUENTIAL DAMAGES FOR BREACH OF ANY EXPRESS OR IMPLIED
WARRANTY.

Some states do not allow the exclusion or limitation of incidental or
consequential damages or limitations on how long an implied warranty
lasts, so the above limitation or exclusion may not apply to you. This
warranty gives you specific legal rights, and you may also have other
rights which vary from state to state. Address all correspondence to:
Virgin Mastertronic Intl., 711 W. 17th Street, Ste. G9, Costa Mesa, CA
92627. Or call (714) 631-1001.

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=query_text)

    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    print(response_text)
    return response_text


if __name__ == "__main__":
    main()
