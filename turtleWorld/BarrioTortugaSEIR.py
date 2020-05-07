from mesa import Model
from mesa.space import MultiGrid, NetworkGrid
from mesa.datacollection import DataCollector
from mesa import Agent
from mesa.time import RandomActivation
import numpy as np

from scipy.stats import gamma
from scipy.stats import expon
from . stats import c19_nbinom_rvs

from . turtle_functions import number_of_infected
from . turtle_functions import number_of_susceptible
from . turtle_functions import number_of_recovered
from . turtle_functions import number_of_exposed

from . utils import PrtLvl, print_level, throw_dice, in_range

CALIB = False
prtl=PrtLvl.Concise

def get_time(t_dist, t_mean):
    if t_dist == 'E':
        return expon.rvs(scale=t_mean)
    elif t_dist == 'G':
        return gamma.rvs(a=t_mean, scale=1.0)
    else:
        return t_mean

class BarrioTortugaBase(Model):
    """Base class for Turtle models of SEIR epidemics.

        The parameters are:
            turtles: the number of agents in the simulation.
            i0     : the initial number of infected agents.
            r0     : basic reproductive number
            p      : transmission probability per contact
            ti     : incubation time
            tr     : recovery time = duration of infectiouness.

        Stochastic behaviour can be introduced in:
        1) ti and tr, which can be chosen to be either gamma or exponentially distributed
        2) p, via R0, which can be chosen to be either negative binomial or Poisson.

        This is controlled through parameters ti_dist, tr_dist and p_dist

    """
    def __init__(self,
                 ticks_per_day =    5,
                 i0            =   10,
                 r0            =    3.5,
                 ti            =    5.5,
                 tr            =    6.5,
                 ti_dist       =    'F',    # F for fixed, E for exp G for Gamma
                 tr_dist       =    'F',
                 p_dist        =    'F'     # F for fixed, S for Binomial, P for Poissoin
                 ):


        self.ticks_per_day = ticks_per_day
        self.i0            = i0
        self.r0            = r0
        self.ti            = ti
        self.tr            = tr
        self.ti_dist       = ti_dist
        self.tr_dist       = tr_dist
        self.p_dist        = p_dist
        self.P        = []
        self.Ti       = []
        self.Tr       = []

        self.schedule   = RandomActivation(self)

        # Prob
        self.k = 1
        if   self.p_dist == 'S':
            self.k  = 0.16
        elif self.p_dist == 'P':
            self.k = 1e+4

        self.datacollector          = DataCollector(
        model_reporters             = {"NumberOfInfected": number_of_infected,
                                       "NumberOfSusceptible": number_of_susceptible,
                                       "NumberOfRecovered": number_of_recovered,
                                       "NumberOfExposed": number_of_exposed}
            )

        if print_level(prtl, PrtLvl.Concise):
            print(f""" Simulation Parameters:

            General
                initial infected        = {self.i0}
                ticks per day           = {self.ticks_per_day}

            Control of stochastics
                ti_dist = {self.ti_dist}
                tr_dist = {self.tr_dist}
                p_dist =  {self.ti_dist}

            Average parameters controlling t and prob

                ti     =  {self.ti}
                tr     =  {self.tr}
                k      =  {self.k}
                r0     =  {self.r0}

            """)


    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)


    def get_prob(self):
        if self.p_dist == 'S' or self.p_dist == 'P':
            r0  = c19_nbinom_rvs(self.r0, self.k) # self.k decided which one
            p   = r0 /(self.nc * self.tr * self.ticks_per_day)
        else:
            p = self.p
        return p


    def infection_prob(self, nc):
        # infection probability for fixed case
        return self.r0 /(nc * self.tr * self.ticks_per_day)


    def print_gen_simul_params(self):
        print(f""" Simulation Parameters:

        General
            number of turtles       = {self.turtles}
            initial infected        = {self.i0}
            ticks per day           = {self.ticks_per_day}

        Control of stochastics
            ti_dist = {self.ti_dist}
            tr_dist = {self.tr_dist}
            p_dist =  {self.ti_dist}

        Average parameters controlling t and prob

            ti     =  {self.ti}
            tr     =  {self.tr}
            k      =  {self.k}
            r0     =  {self.r0}

        Number of contacts and infection prob
            nc     =  {self.nc}
            p      =  {self.p}


        """)

class BarrioTortugaSEIR(BarrioTortugaBase):
    """A simple model of SEIR epidemics.

    The parameters are:
        turtles: the number of agents in the simulation.


        Model calibration. R0 can be written as:

        R0 = c x p x ti

        where:
           c is the average number of contacts per unit time
           c = nc x N /a
           where, for large density :
           nc = 9 (number of cells in Moore)
           N  = number of turtles
           a  = area of grid (w x h)

           p is the transmission probability per contact
           ti is the duration of the infection (the time the turtle is in the I cathegory)

       Two quantities can be taken as known, R0 and ti. Then one can determine p as:
           p = R0 /(c * ti)


    """

    def __init__(self,
                 ticks_per_day =    5,
                 turtles       = 1000,
                 i0            =   10,
                 r0            =    3.5,
                 ti            =    5.5,
                 tr            =    3.5,
                 ti_dist       =    'F',    # F for fixed, E for exp G for Gamma
                 tr_dist       =    'F',
                 p_dist        =    'F',    # F for fixed, S for Binomial, P for Poissoin
                 width         =   40,
                 height        =   40):

        super().__init__(ticks_per_day, i0, r0, ti, tr, ti_dist, tr_dist, p_dist)

        # define grid and schedule
        self.height     = height
        self.width      = width
        self.grid       = MultiGrid(self.height, self.width, torus=True)
        self.moore      = True  # always 9 cells
        self.turtles  = turtles

        # average number of contacts:  nc = 9 * N / area
        # where N / area is the average population per cell and 9 the number of cells
        self.nc         = 9 * self.turtles / (self.width * self.height)

        # infection probability for fixed case
        self.p   = self.infection_prob(self.nc)

        if print_level(prtl, PrtLvl.Concise):
            self.print_gen_simul_params()
            print(f""" Additional Simulation Parameters:
                Grid (w x h)            = {self.width} x {self.height}
            """)


        # Create turtles
        if CALIB:  # only susceptible agents
            for i in range(self.turtles):
                x,y = self.random_pos()           # random position
                a = SeirTurtle(i, (x, y), 'S', ti, tr, 1, self)
                self.schedule.add(a)              # add to schedule
                self.grid.place_agent(a, (x, y))  # added to schedule

        else:
            ss = self.turtles - i0            # number of susceptibles
            A = ss * ['S'] + i0 * ['I']
            np.random.shuffle(A)              # in random order

            for i, at in enumerate(A):
                x,y = self.random_pos()           # random position
                ti = get_time(self.ti_dist, self.ti)
                #print(f'from exp  ti ={ti}')
                tr = get_time(self.tr_dist, self.tr)
                p  = self.get_prob()
                self.Ti.append(ti)
                self.Tr.append(tr)
                self.P.append(p)

                if print_level(prtl, PrtLvl.Concise):
                    if i < 5:
                        print (f' creating turtle number {i} with ti = {ti}, tr = {tr}, p ={p:.2e}')

                if at == 'I':
                    if print_level(prtl, PrtLvl.Concise):
                        if i < 5:
                            print (f' creating I turtle')

                    a = SeirTurtle(i, (x, y), 'I',
                                   ti * ticks_per_day,
                                   tr * ticks_per_day, p,
                                   self)

                else:
                    if print_level(prtl, PrtLvl.Concise):
                        if i < 5:
                            print (f' creating S turtle')

                    a = SeirTurtle(i, (x, y), 'S',
                                   ti * ticks_per_day,
                                   tr * ticks_per_day, p,
                                   self)

                self.schedule.add(a)              # add to schedule
                self.grid.place_agent(a, (x, y))  # added to schedule

        self.running = True
        self.datacollector.collect(self)


    def random_pos(self):
        x = self.random.randrange(self.width)
        y = self.random.randrange(self.height)
        return x, y


class BarrioTortugaNX(BarrioTortugaBase):
    """A simple model of  epidemics using networx.

    The parameters are:
        G : the network chosen
        neighbors: average number of neighbors (nodes) per turle (node)

    """


    def __init__(self,
                 G,
                 neighbors,
                 ticks_per_day =    5,
                 i0            =   10,
                 r0            =    3.5,
                 ti            =    5.5,
                 tr            =    6.5,
                 ti_dist       =    'F',    # F for fixed, E for exp G for Gamma
                 tr_dist       =    'F',
                 p_dist        =    'F',    # F for fixed, S for Binomial, P for Poissoin
                 ):

        super().__init__(ticks_per_day, i0, r0, ti, tr, ti_dist, tr_dist, p_dist)
        # define grid and schedule

        self.grid = NetworkGrid(G)
        self.turtles  = len(G)
        self.nc         = neighbors

        # infection probability for fixed case
        self.p   = self.infection_prob(self.nc)
        if print_level(prtl, PrtLvl.Concise):
            self.print_gen_simul_params()


        # Create turtles
        ss = self.turtles - i0            # number of susceptibles
        A = ss * ['S'] + i0 * ['I']
        np.random.shuffle(A)              # in random order

        for i, at in enumerate(A):
            ti = get_time(self.ti_dist, self.ti)
            tr = get_time(self.tr_dist, self.tr)
            p  = self.get_prob()

            self.Ti.append(ti)
            self.Tr.append(tr)
            self.P.append(p)

            if print_level(prtl, PrtLvl.Concise):
                if i < 5:
                    print (f' creating turtle number {i} with ti = {ti}, tr = {tr}, p ={p:.2e}')

            if at == 'I':
                if print_level(prtl, PrtLvl.Concise):
                    if i < 5:
                        print (f' creating I turtle')

                a = NXTurtle(i,  'I',
                                   ti * ticks_per_day,
                                   tr * ticks_per_day, p,
                                   self)

            else:
                if print_level(prtl, PrtLvl.Concise):
                    if i < 5:
                        print (f' creating S turtle')

                a = NXTurtle(i,  'S',
                                   ti * ticks_per_day,
                                   tr * ticks_per_day, p,
                                   self)

            self.schedule.add(a)              # add to schedule
            self.grid.place_agent(a, i)       # added to grid

        self.running = True
        self.datacollector.collect(self)


class TurtleBase(Agent):
    '''
    Base Class for a turtle

    '''

    def __init__(self, unique_id, kind, ti, tr, prob, model):
        super().__init__(unique_id, model)

        self.p    = prob
        self.kind = kind
        self.ti   = ti           # equals model average for now throw dist later
        self.tr   = tr           # equals model average for now throw dist later
        self.il   = 0            # counter tick
        self.iil  = 0           # infection length
        self.iel  = 0           # infection length


    def infection_step(self):
        self.il+=1

        # Turtle became exposed with tag self.iel (see infect ())
        if self.kind == 'E' :

            if print_level(prtl, PrtLvl.Detailed):
                    print(f"""Found exposed with tag = {self.iel}
                          global time = {self.model.schedule.steps}
                          turtle id   = {self.unique_id}
                """)

            # When time is larger than incubation time, become infected
            if self.model.schedule.steps - self.iel > self.ti :
                self.iil = self.model.schedule.steps
                self.kind = 'I'

                if print_level(prtl, PrtLvl.Detailed):
                    print(f"""Turning E into I with tag = {self.iil}
                          global time = {self.model.schedule.steps}
                          turtle id   = {self.unique_id}
                """)

        elif self.kind == 'I':
            if print_level(prtl, PrtLvl.Detailed):
                print(f"""Found Infected with tag = {self.iil}
                          global time = {self.model.schedule.steps}
                          turtle id   = {self.unique_id}
                          **going to infect***
                """)

            self.infect()

            # When time is larger than recovery time, become recovered
            if self.model.schedule.steps - self.iil >  self.tr :
                self.kind = 'R'

                if print_level(prtl, PrtLvl.Detailed):
                    print(f"""Turning I into R with tag = {self.iil}
                          global time = {self.model.schedule.steps}
                          turtle id   = {self.unique_id}
                """)


    def infect(self):
        pass


    def turning_into_exposed(self, turtle):
        if print_level(prtl, PrtLvl.Verbose):
            print(f' throwing dice')

        if throw_dice(self.p):
            turtle.kind = 'E'
            turtle.iel = self.model.schedule.steps # tag = infection time

            if print_level(prtl, PrtLvl.Detailed):
                print(f' **TURNING TURTLE INTO E ** ')
                print(f"""tag  {turtle.iel}
                          global time = {self.model.schedule.steps}
                          turtle id   = {turtle.unique_id}
                          turtle kind = {turtle.kind}

                """)


    def print_infection_banner(self):
        print(f"""Now infecting with tags  {self.iil}
                  global time = {self.model.schedule.steps}
                  turtle id   = {self.unique_id}

        """)


class SeirTurtle(TurtleBase):
    '''
    Class implementing a SEIR turtle that can move at random

    '''

    def __init__(self, unique_id, pos, kind, ti, tr, prob, model):
        '''
        grid: The MultiGrid object in which the agent lives.
        x: The agent's current x coordinate
        y: The agent's current y coordinate
        stochastic: If false mean average
        '''
        super().__init__(unique_id, kind, ti, tr, prob, model)
        self.pos  = pos


    def step(self):
        self.infection_step()
        self.random_move()


    def infect(self):
        if print_level(prtl, PrtLvl.Verbose):
            self.print_infection_banner()

        # array of coordinates ((x0,y0), (x1,y1), (x2, y2),...) of neighbors
        # last parameter True inludes own cell
        n_xy = self.model.grid.get_neighborhood(self.pos, self.model.moore, True)

        if print_level(prtl, PrtLvl.Verbose):
                print(f'coordinates of neighbors, including me = {n_xy}')

        for xy in n_xy:   # loops over all cells
            if print_level(prtl, PrtLvl.Verbose):
                    print(f'neighbors = {xy}')

            cell = self.model.grid.get_cell_list_contents(xy)
            turtles = [obj for obj in cell if isinstance(obj, type(self))]

            if print_level(prtl, PrtLvl.Verbose):
                print(f' number of turtles = {len(turtles)}')

            for turtle in turtles:  # loops over all turtles in cells

                if print_level(prtl, PrtLvl.Verbose):
                    print(f' turtle kind = {turtle.kind}')

                if turtle.kind == 'S':  # if susceptible found try to infect
                    self.turning_into_exposed(turtle)


    def random_move(self):
        '''
        Step one cell in any allowable direction.
        We use grid’s built-in get_neighborhood method, which returns all the neighbors of a given cell.
        This method can get two types of cell neighborhoods: Moore (including diagonals),
        and Von Neumann (only up/down/left/right).
        It also needs an argument as to whether to include the center cell itself as one of the neighbors.
        '''
        # Pick the next cell from the adjacent cells.
        next_moves = self.model.grid.get_neighborhood(self.pos, self.model.moore, True)
        next_move = self.random.choice(next_moves)
        # Now move:
        self.model.grid.move_agent(self, next_move)


class NXTurtle(TurtleBase):
    '''
    Class implementing a SEIR turtle living in a network

    '''

    def __init__(self, unique_id, kind, ti, tr, prob, model):
        '''
        grid: The MultiGrid object in which the agent lives.
        x: The agent's current x coordinate
        y: The agent's current y coordinate
        stochastic: If false mean average
        '''
        super().__init__(unique_id, kind, ti, tr, prob, model)
        self.pos   = unique_id   # node


    def step(self):
        self.infection_step()


    def infect(self):
        if print_level(prtl, PrtLvl.Verbose):
            self.print_infection_banner()

        neighbors = self.model.grid.get_neighbors(self.pos) # neighbor nodes

        if print_level(prtl, PrtLvl.Verbose):
                print(f'neighbors  = {neighbors}')

        turtles = self.model.grid.get_cell_list_contents(neighbors)

        for turtle in turtles:  # loops over all turtles in cells

            if print_level(prtl, PrtLvl.Verbose):
                print(f' turtle kind = {turtle.kind}')

            if turtle.kind == 'S':  # if susceptible found try to infect
                self.turning_into_exposed(turtle)
