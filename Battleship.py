import numpy as np
import matplotlib.pyplot as plt

GRID_SIZE = 10

# Boats Identifications
SHIP_IDS =\
{
	"Aircraft carrier" : 1,
	"Cruiser"          : 2,
	"Anti-Destroyer"   : 3,
	"Submarin"         : 4,
	"Destroyer"        : 5
}
SHIP_LENGTHS =\
{
	1: 5,
	2: 4,
	3: 3,
	4: 3,
	5: 2
}


class Game:
	# Instantiate a new Object
	def __init__(self):
		self.grid_IA_1 = Battlefield()
		self.grid_IA_2 = Battlefield()

	# Randomely generate two grid
	def random(self):
		self.grid_IA_1.random_grid()
		self.grid_IA_2.random_grid()

	def show(self, show: bool = False, debug: bool = False)->None:
		"""
		Show the grids, at least one of two parameters must be True
		Parameter:
			show  : to show  the grids as images
			debug : to print the grids as matrixes
		"""
		print("\nGrille du IA 1:")
		if   show : self.grid_IA_1.show("A")
		elif debug: print(self.grid_IA_1.grid)
		else:
			return; print("[show] At least one of two parameters must be True!")

		print("\nGrille du IA 2:")
		if show :self.grid_IA_2.show("B")
		if debug: print(self.grid_IA_2.grid)

class Battlefield:

	# Instantiate a new Object
	def __init__(self):
		self.grid : np.matrix = np.zeros((10, 10), dtype=int)

		self.boats: Dict[str,list[tuple[int,int]]] =\
		{
			1 : [],
			2 : [],
			3 : [],
			4 : [],
			5 : []
		}

	def is_empty(
		self, boat: int, position: tuple[int,int], direction: str
	)->bool:
		"""
		Check if their is space for the boat
		Parameters:
			boat      : integer from SHIP_IDS
			position  : tuple of x,y coordinate
			direction : of the boat, either vertical or horizontal
			debug     : print debug information
		Returns:
			True if their is space, false otherwise
		"""

		x,y    = position
		length = SHIP_LENGTHS[boat]

		if direction == "horizontal":

			if y +length > GRID_SIZE: return False

			for i in range(length):
				if self.grid[x,y +i] != 0: return False

		elif direction == "vertical":
			if x +length > GRID_SIZE: return False

			for i in range(length):
				if self.grid[x +i,y] != 0: return False
		else:
			print("[is_empty] Bad direction value error!")
			return False
		return True

	def place(
		self, boat: int, position: tuple[int,int], direction: str
	)->bool:
		"""
		Place a boat at position if their is space
		Parameters:
			boat      : integer from SHIP_IDS
			position  : tuple of x,y coordinate
			direction : of the boat, either vertical or horizontal
		Return:
			A matrix with the boat added on it
		"""
		x,y = position; length = SHIP_LENGTHS[boat]

		if not self.is_empty(boat, position, direction): return False

		if direction == "horizontal":
			for i in range(length):
				self.boats[SHIP_IDS[boat]].append( (x,y +i) )
				self.grid[x,y +i] = boat

		elif direction == "vertical":
			for j in range(length):
				self.boats[SHIP_IDS[boat]].append( (x +j,y) )
				self.grid[x +j,y] = boat
		return True

	# Randomly places boats on the grid
	def random_grid(self)->None:
		for boat in SHIP_IDS.values():
			while True:
				x = np.random.randint(0, GRID_SIZE)
				y = np.random.randint(0, GRID_SIZE)
				direction = np.random.choice(["horizontal", "vertical"])

				if self.place(boat, (x, y), direction): break

	def show(self, title: str)->None:
		"""
		Print a visual of the grid
		Parameter:
			title : name of the graph
		"""
		plt.figure()
		plt.imshow(self.grid, aspect="equal", cmap="viridis", origin="upper")
		plt.axis('off')
		plt.title(title)
		plt.show()

	def __equal__(gridA: np.matrix, gridB: np.matrix)->bool:
		"""
		Static fuction
		Compare to grid if their equals
		Return: True if their are equals, false otherwise
		"""
		return np.array_equal(gridA, gridB)

	def play(self, position: tuple[int,int])->bool:
		"""
		Launch a torpedo
		Parameter:
			position : the coordinate of the attack
		Return:
			True if the hit is valid, False otherwise
		"""
		x,y = position
		temp = self.grid[x,y]
		if temp != 0:
			if temp == 9:
				print("Already hit!!!"); return False

			self.grid[x,y] = 9
			self.boats[temp].find(position).remove()

			if self.boats[temp].is_empty():

				print("Sunked!"); return True
			else:
				print("Hit!");    return True

	# Test if all your ship are sunked
	def victory(self)->None:
		for boat in slef.boats:
			if not boat.is_empty: return
		print("You have lost the game!!!")

	def reset(self): self.grid = np.zeros((10, 10), dtype=int)


class IA_:
	def __init__(self): self.score: int = 0

	def random_IA(self): pass
	def heuristic_IA(self): pass
	def simple_propability_IA(self): pass
	def Monte_Carlo_IA(self): pass

game = Game();
game.random()
# game.show(True)


# Tested
# Add debug param->Debug output at function call
# Game class
# Battlefield class
#   equals
#   play
#   victory
#   reset
# IA class
#   random
#   heuristic
#   simple prob°
#   Monte carlos ?