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

        self.boats =\
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
                self.boats[boat].append( (x,y +i) )
                self.grid[x,y +i] = boat

        elif direction == "vertical":
            for j in range(length):
                self.boats[boat].append( (x +j,y) )
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

    def play(self, position: tuple[int,int])->int:
        """
        Launch a torpedo
        Parameter:
            position : the coordinate of the attack
        Return:
            True if the hit is valid, False otherwise
        """
        x,y = position
        temp = self.grid[x,y]
        if temp == 9:
           print("Already hit!!!"); 
           return -1
        
        if temp != 0 and temp != 9:

            self.grid[x,y] = 9
            self.boats[temp].remove(position)

            if not self.boats[temp]:

                print("Sunked!"); return 1
            else:
                print("Hit!");    return 2
                
        else :
            self.grid[x,y] = 9
            print("MISS")
            return 0
                
                
                

    # Test if all your ship are sunked
    def victory(self)->bool:
        for boat in self.boats.values():
            if  boat: 
                return False
            
            
        print("Win")
        return True
    
    def reset(self): 
        self.grid = np.zeros((10, 10), dtype=int)


class random_IA:
    def __init__(self): 
        self.score: int = 0
        self.grid_IA=Battlefield()
        self.grid_IA.random_grid()
        
    def random_IA(self): 
        fire=0
        pos_possible=[(i,j) for i in range(0,GRID_SIZE) for j in range(0,GRID_SIZE)]
        while not self.grid_IA.victory() :
            x,y=np.random.randint(0,(GRID_SIZE)),np.random.randint(0,(GRID_SIZE))
            
            if (x,y) in pos_possible :
                self.grid_IA.play((x,y))
                fire +=1
                pos_possible.remove((x,y))

        print(fire)
        return fire
    
    
    
class heuristic_IA:
    def __init__(self): 
        self.score: int = 0
        self.grid_IA=Battlefield()
        self.grid_IA.random_grid()
    
    def heuristic_IA(self): 
        fire=0
        pos_possible=[(i,j) for i in range(0,GRID_SIZE) for j in range(0,GRID_SIZE)]
        boat_hit=[]
        while not self.grid_IA.victory():
            if (len(boat_hit)==0) :
                x,y=np.random.randint(0,GRID_SIZE),np.random.randint(0,GRID_SIZE)
                if (x,y) in pos_possible :
                    if self.grid_IA.grid[x,y] !=0 and self.grid_IA.grid[x,y] !=9:
                        self.grid_IA.play((x,y))
                        boat_hit.append((x,y))
                        fire +=1
                        pos_possible.remove((x,y))
                    else :
                        self.grid_IA.play((x,y))
                        fire +=1
                        pos_possible.remove((x,y))
            else :
                px,py=boat_hit[0]
                pos_connexe=[(px+1,py),(px-1,py),(px,py+1),(px,py-1)]
                boat_hit.pop(0)
                for pos in pos_connexe :
                    if pos in pos_possible :
                        if self.grid_IA.grid[pos] !=0 and self.grid_IA.grid[pos] !=9:
                            self.grid_IA.play(pos)
                            fire +=1
                            pos_possible.remove(pos)
                            boat_hit.append(pos)
                        else :
                            self.grid_IA.play(pos)
                            fire +=1
                            pos_possible.remove(pos)
        print(fire)
        return fire
    
    
    
class simple_propability_IA:
    def __init__(self): 
        self.score: int = 0
        self.grid_IA=Battlefield()
        self.grid_IA.random_grid()
    
    
    def peut_placer_prob(self,grid_prob, boat, position, direction):
        Px, Py = position
        length = SHIP_LENGTHS[boat]
        if((Px >= GRID_SIZE) or (Py >= GRID_SIZE) or (Px <0) or (Py <0)):
            return False

        if(boat > 5 or boat < 1):
            return False
        else:
            if(direction == "horizontal"):
                if(Py+length > len(grid_prob)):
                    return False
                else:
                    i = 0
                    while(i < length):
                        if(grid_prob[Px, Py+i] == -1):
                            return False
                        i += 1
                    return True
            else:
                if(Px+length > len(grid_prob)):
                    return False
                else:
                    i = 0
                    while(i < length):
                        if(grid_prob[Px+i, Py] == -1):
                            return False
                        i += 1
                    return True
        
    
    def calcu_proba_simple(self,list_boat,grid_prob) :
        prob_max=0
        pos_max=(0,0)
        for x in range(0,GRID_SIZE):
            for y in range(0,GRID_SIZE):
                if grid_prob[x,y] != -1:
                    grid_prob[x,y]=0
        for boat in list_boat:
            length = SHIP_LENGTHS[boat]
            for x in range(0,GRID_SIZE):
                for y in range(0,GRID_SIZE):
                    if(self.peut_placer_prob(grid_prob,boat,(x,y),"horizontal")) :
                        for i in range(length):
                            grid_prob[x,y+i]+=1
                            if(grid_prob[x,y+i]>prob_max):
                                prob_max=grid_prob[x,y+i]
                                pos_max=(x,y+i)
                                
                    if(self.peut_placer_prob(grid_prob,boat,(x,y),"vertical")) :
                        for i in range(length):
                            grid_prob[x+i,y]+=1
                            if(grid_prob[x+i,y]>prob_max):
                                prob_max=grid_prob[x+i,y]
                                pos_max=(x+i,y)

                                
        for x in range(0,GRID_SIZE):
            for y in range(0,GRID_SIZE):
                if(grid_prob[x,y] >= 0):
                    grid_prob[x,y]
        return pos_max
        
    def simple_propability_IA(self): 
        liste_boats=[1,2,3,4,5]
        fire=0
        pos_possible=[(i,j) for i in range(0,GRID_SIZE) for j in range(0,GRID_SIZE)]
        grid_prob=np.zeros((GRID_SIZE,GRID_SIZE))
        boat_hit=[]
        while not self.grid_IA.victory() :
            if (len(boat_hit)==0) :
                pos_max=self.calcu_proba_simple(liste_boats,grid_prob)
                if pos_max in pos_possible :
                    if self.grid_IA.grid[pos_max] !=0 and self.grid_IA.grid[pos_max] !=9 :
                        boat_hit.append(pos_max)
                        temp=self.grid_IA.grid[pos_max]
                        tmp=self.grid_IA.play(pos_max)
                        fire += 1
                        pos_possible.remove(pos_max)
                        grid_prob[pos_max]=-1

                        if(tmp == 1):
                            liste_boats.remove(temp)
                            

                    else:
                        self.grid_IA.play(pos_max)

                        fire += 1
                        pos_possible.remove(pos_max)
                        grid_prob[pos_max]=-1
                        
            else :
                px,py=boat_hit[0]
                pos_connexe=[(px+1,py),(px-1,py),(px,py+1),(px,py-1)]
                boat_hit.pop(0)
                for pos in pos_connexe :
                    if pos in pos_possible :
                        if self.grid_IA.grid[pos] !=0 and self.grid_IA.grid[pos] !=9:
                            temp=self.grid_IA.grid[pos]
                            tmp=self.grid_IA.play(pos)
                            fire +=1
                            pos_possible.remove(pos)
                            grid_prob[pos]=-1
                            boat_hit.append(pos)
                            if(tmp == 1):
                                liste_boats.remove(temp)
                        else :
                            self.grid_IA.play(pos)
                            fire +=1
                            pos_possible.remove(pos)
                            grid_prob[pos]=-1
        print(fire)
        return fire
    
    
    
class Monte_Carlo_IA:
    def __init__(self): 
        self.score: int = 0
        self.grid_IA=Battlefield()
        self.grid_IA.random_grid()
    
    
    def peut_placer_prob_grid(self,grid_prob,grid_temp, boat, position, direction):
        Px, Py = position
        length = SHIP_LENGTHS[boat]
        if((Px >= GRID_SIZE) or (Py >= GRID_SIZE) or (Px <0) or (Py <0)):
            return False

        if(boat > 5 or boat < 1):
            return False
        else:
            if(direction == "horizontal"):
                if(Py+length > len(grid_prob) or Py+length > len(grid_temp) ):
                    return False
                else:
                    i = 0
                    while(i < length):
                        if(grid_prob[Px, Py+i] == -1 or grid_temp[Px, Py+i] != 0):
                            return False
                        i += 1
                    return True
            else:
                if(Px+length > len(grid_prob) or Px+length > len(grid_temp)):
                    return False
                else:
                    i = 0
                    while(i < length):
                        if(grid_prob[Px+i, Py] == -1 or grid_temp[Px+i, Py] != 0):
                            return False
                        i += 1
                    return True
    
    
                
    def calcu_proba_Monte_Carlo(self,list_boat,grid_prob,grid_temp,N) :
            
        
        for x in range(0,GRID_SIZE):
            for y in range(0,GRID_SIZE):
                if grid_prob[x,y] != -1:
                    grid_prob[x,y]=0
        
        for i in range(N) :
            for x in range(0,GRID_SIZE):
                for y in range(0,GRID_SIZE):
                    if grid_temp[x,y] == 8:
                        grid_temp[x,y]=0
            
            prob_max=0
            pos_max=(0,0)
            for boat in list_boat:
                length = SHIP_LENGTHS[boat]
                while True :
                    
                    x = np.random.randint(0, GRID_SIZE)
                    y = np.random.randint(0, GRID_SIZE)
                    direction = np.random.choice(["horizontal", "vertical"])
                    if(self.peut_placer_prob_grid(grid_prob,grid_temp,boat,(x,y),direction)) :
                        if(direction=="horizontal"):
                                for i in range(length) :
                                    grid_temp[x,y+i]=8

                                    grid_prob[x,y+i]+=1
                                    if(grid_prob[x,y+i]>prob_max):
                                        prob_max=grid_prob[x,y+i]
                                        pos_max=(x,y+i)
                                    
                        else:
                                for i in range(length) :
                                
                                    grid_temp[x+i,y]=8
                                    grid_prob[x+i,y]+=1
                                    if(grid_prob[x+i,y]>prob_max):
                                        prob_max=grid_prob[x+i,y]
                                        pos_max=(x+i,y)
                        break
        for x in range(0,GRID_SIZE):
            for y in range(0,GRID_SIZE):
                if(grid_prob[x,y] >= 0):
                    grid_prob[x,y] 

        return pos_max
            
    def Monte_Carlo_IA(self): 
        liste_boats=[1,2,3,4,5]
        fire=0
        pos_possible=[(i,j) for i in range(0,GRID_SIZE) for j in range(0,GRID_SIZE)]
        grid_prob=np.zeros((GRID_SIZE,GRID_SIZE))
        grid_temp=np.zeros((GRID_SIZE,GRID_SIZE))
        boat_hit=[]
        while not self.grid_IA.victory() :
            if (len(boat_hit)==0) :
                pos_max=self.calcu_proba_Monte_Carlo(liste_boats,grid_prob,grid_temp,20)
                if pos_max in pos_possible :
                    if self.grid_IA.grid[pos_max] !=0 and self.grid_IA.grid[pos_max] !=9 :
                        boat_hit.append(pos_max)
                        temp=self.grid_IA.grid[pos_max]
                        tmp=self.grid_IA.play(pos_max)
                        fire += 1
                        pos_possible.remove(pos_max)
                        grid_prob[pos_max]=-1
                        grid_temp[pos_max]=9

                        if(tmp == 1):
                            liste_boats.remove(temp)
                            

                    else:
                        self.grid_IA.play(pos_max)

                        fire += 1
                        pos_possible.remove(pos_max)
                        grid_prob[pos_max]=-1
                        grid_temp[pos_max]=9
                        
            else :
                px,py=boat_hit[0]
                pos_connexe=[(px+1,py),(px-1,py),(px,py+1),(px,py-1)]
                boat_hit.pop(0)
                for pos in pos_connexe :
                    if pos in pos_possible :
                        if self.grid_IA.grid[pos] !=0 and self.grid_IA.grid[pos] !=9:
                            temp=self.grid_IA.grid[pos]
                            tmp=self.grid_IA.play(pos)
                            fire +=1
                            pos_possible.remove(pos)
                            grid_prob[pos]=-1
                            grid_temp[pos_max]=9
                            boat_hit.append(pos)
                            if(tmp == 1):
                                liste_boats.remove(temp)
                        else :
                            self.grid_IA.play(pos)
                            fire +=1
                            pos_possible.remove(pos)
                            grid_prob[pos]=-1
                            grid_temp[pos_max]=9
        print(fire)
        return fire

game = Game()
game.random()
game.show(True,True)


distribution_aleatoire1=[]
for i in range(1,1001):
  joueur1 = random_IA()
  compteur1=joueur1.random_IA()
  distribution_aleatoire1.append(compteur1)
plt.hist(distribution_aleatoire1)
plt.xlabel('nombre de cout')
plt.ylabel("nombre d'occurence")
plt.show()

distribution_aleatoire2=[]
for i in range(1,1001):
  joueur2 = heuristic_IA()
  compteur2=joueur2.heuristic_IA()
  distribution_aleatoire2.append(compteur2)
plt.hist(distribution_aleatoire2)
plt.xlabel('nombre de cout')
plt.ylabel("nombre d'occurence")
plt.show()

distribution_aleatoire3=[]
for i in range(1,1001):
  joueur3 = simple_propability_IA()
  compteur3=joueur3.simple_propability_IA()
  distribution_aleatoire3.append(compteur3)
plt.hist(distribution_aleatoire3)
plt.xlabel('nombre de cout')
plt.ylabel("nombre d'occurence")
plt.show()


distribution_aleatoire4=[]
for i in range(1,1001):
  joueur4 = Monte_Carlo_IA()
  compteur4=joueur4.Monte_Carlo_IA()
  distribution_aleatoire4.append(compteur4)
plt.hist(distribution_aleatoire4)
plt.xlabel('nombre de cout')
plt.ylabel("nombre d'occurence")
plt.show()

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

