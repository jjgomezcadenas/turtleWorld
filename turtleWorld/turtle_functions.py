
def number_turtles_in_cell(cell):
    turtles = [obj for obj in cell if isinstance(obj, SeirTurtle)]
    return len(turtles)


def turtles_in_cell(cell):
    if number_turtles_in_cell(cell) > 0:
        return True
    else:
        return False


def number_of_infected(model):
    a =[agent for agent in model.schedule.agents if agent.kind=='I']
    return len(a)


def number_of_susceptible(model):
    a =[agent for agent in model.schedule.agents if agent.kind=='S']
    return len(a)


def number_of_recovered(model):
    a =[agent for agent in model.schedule.agents if agent.kind=='R']
    return len(a)


def number_of_exposed(model):
    a =[agent for agent in model.schedule.agents if agent.kind=='E']
    return len(a)


def number_of_turtles_in_neighborhood(model):
    nc = 0
    ng = 0
    NC = []
    for y in range(model.grid.height):
        for x in range(model.grid.width):
            ng +=1

            c = model.grid.get_cell_list_contents((x,y))
            n_turtles_in_c = number_turtles_in_cell(c)

            if n_turtles_in_c > 0:
                #coordinates of neighbors
                n_xy = model.grid.get_neighborhood((x,y), model.moore, True)

                ncc = 0
                for xy in n_xy:
                    cn = model.grid.get_cell_list_contents(xy)
                    n_turtles_nb = number_turtles_in_cell(cn)

                    if n_turtles_nb > 0:
                        nc += n_turtles_nb
                        ncc += n_turtles_nb
                NC.append(ncc)

    return np.mean(NC)
