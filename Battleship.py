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

class Game: pass

class Battlefield:

	def __init__(self):
		self.grid: np.matrix = np.zeros((10, 10), dtype=int)

	def is_empty(
		self, boat: int, position: tuple[int,int], direction: str
	)->bool:
		"""
		Check if their space for the boat
		Parameters:
			boat      : integer from SHIP_IDS
			position  : tuple of x,y coordinate
			direction : of the boat, either vertical or horizontal
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
		x,y    = position
		length = SHIP_LENGTHS[boat]
		print("Debugg: ", self.is_empty(boat, position, direction), position)

		if not self.is_empty(boat, position, direction): return False

		if direction == "horizontal":
			for i in range(length):
				print(x,y)
				self.grid[x,y +i] = boat

		elif direction == "vertical":
			for j in range(length):
				print(x,y)
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

	def play(self, position): pass
	def victory(self): pass
	def reset(self): pass

class IA_:
	def __init__(self): self.score: int = 0

	def random_IA(self): pass
	def heuristic_IA(self): pass
	def simple_propability_IA(self): pass
	def Monte_Carlo_IA(self): pass

# Générer une grid pour chaque IA
grid_IA_1 = Battlefield(); grid_IA_1.random_grid()
grid_IA_2 = Battlefield(); grid_IA_2.random_grid()

# Afficher les grids des deux IA_s
print("Grille du IA 1:")
grid_IA_1.show("A")
print(grid_IA_1.grid)

print("Grille du IA 2:")
grid_IA_2.show("B")
print(grid_IA_2.grid)