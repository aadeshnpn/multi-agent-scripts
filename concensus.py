import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

random_seed = 123
random = np.random.RandomState(random_seed)

"""
* Randomly place agents in a two dimensional world.  Place the agents at uniformly random x locations in the range $[-M,M]$ and similary for random y locations.
* Create the adjacency graph that results when an agent is connected to all agents within a fixed metric distance, R.
* What is the probability that the resulting graph is connected as a function of metric distance?
* (Report your results as a function of R/M so that all results from all students are on a similar scale.)
* Create the adjacency graph that results when an agent is connected to its N nearest neighbors.
* What is the probability that the resulting graph is connected as a function of N?  For different values of N,
* what is the average (across some sample of random graphs) distance of the nearest neighbor that is farthest from the agent?
* Which is more likely to be connected, a topological neighborhood or a metric neighborhood?  Justify your answer.
"""

def draw_agents(agents, R=10):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x, y = zip(*agents)
    # ax.scatter(x, y, s=10, alpha=0.75, edgecolors='r')
    ax.plot(x, y, 'ro')
    for i in range(len(x)):
        circle = plt.Circle((x[i], y[i]), R, alpha=0.1, edgecolor='blue', facecolor='red')
        ax.add_artist(circle)
    plt.show()

def draw_distance(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, 'gx')
    ax.set_ylabel('Average Farthest Distance')
    ax.set_xlabel('No. of Nearest Neighbours')
    plt.xlim(0, max(np.array(x)) + 1)
    ax.set_title('Average distance between agents with KNN')
    plt.tight_layout()
    plt.show()

def draw_eigen_values(x, la, lr):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, la,'gx')
    ax.plot(x, lr,'ro')
    ax.set_ylabel('Fiedler Eigen Value')
    ax.set_xlabel('Time step')
    plt.xlim(0, max(np.array(x)))
    ax.set_title('Fiedler Eigne values of agents for simulation')
    plt.tight_layout()
    plt.show(block=True)


def draw_graph(x, y, mode='dist'):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, 'ro')
    ax.set_ylabel('Probability')

    if mode =='dist':
        ax.set_xlabel('R/M')
        ax.set_title('Connectedness Probability of agents')
        plt.xlim(0, max(np.array(x)) + 0.1)
    else:
        ax.set_xlabel('No. of Nearest Neighbours')
        plt.xlim(0, max(np.array(x)) + 1)
        ax.set_title('Connectedness Probability of agents with KNN')

    plt.tight_layout()
    plt.show()

# Create adjacency graph that results when agents are within a fixed metrix distance R
def distance(a, b):
    return np.round(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2), 2)

def create_adjancey_graph(agents, R=10):
    # Compute distance matrix
    n = len(agents)
    D = np.eye(n)
    dist = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i==j:
                dist[i][j] = 200
            else:
                dist[i][j] = distance(agents[i], agents[j])

    a = dist < R
    A = a * 1

    # Compute the degree matrix
    np.fill_diagonal(D, np.sum(A, axis=1))
    L = D - A
    return L, D, A

def create_lalr(agents, a=15, r=10):
    # Compute distance matrix
    n = len(agents)
    Da = np.eye(n)
    Dr = np.eye(n)
    dist = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i==j:
                dist[i][j] = 200
            else:
                dist[i][j] = distance(agents[i], agents[j])

    l1 = dist > r
    l2 = dist < a
    la = np.logical_and(l1, l2)

    Aa = la * 1

    lr = dist < r
    Ar = lr * 1

    # Compute the degree matrix
    np.fill_diagonal(Da, np.sum(Aa, axis=1))
    np.fill_diagonal(Dr, np.sum(Ar, axis=1))

    La = Da - Aa
    Lr = Dr - Ar
    return La, Lr

def create_adjancey_graph_knn(agents, N=3):
    # Compute distance matrix
    n = len(agents)
    D = np.eye(n)
    a = np.zeros(n)
    nbrs = NearestNeighbors(n_neighbors=N, algorithm='auto').fit(agents)
    distances, indices = nbrs.kneighbors(agents)
    # Get the farthest distance from the nearest neighbours
    far_dist = np.mean(list(zip(*distances))[-1])
    # Since this adajency matrix puts 1 to itself. lets remove it
    A = nbrs.kneighbors_graph(agents).toarray()
    np.fill_diagonal(A, a)
    # print (A)
    # Compute the degree matrix
    np.fill_diagonal(D, np.sum(A, axis=1))
    L = D - A
    return L, D, A, np.mean(far_dist)

def create_lalr_knn(agents, ar=2, rr=3):
    # Compute distance matrix
    La, Lr = create_lalr(agents, ar, rr)
    n = len(agents)
    D = np.eye(n)
    a = np.zeros(n)
    nbrs = NearestNeighbors(n_neighbors=ar, algorithm='auto').fit(agents)
    distances, indices = nbrs.kneighbors(agents)
    # Get the farthest distance from the nearest neighbours
    far_dist = np.mean(list(zip(*distances))[-1])
    # Since this adajency matrix puts 1 to itself. lets remove it
    A = nbrs.kneighbors_graph(agents).toarray()
    np.fill_diagonal(A, a)
    # print (A)
    # Compute the degree matrix
    np.fill_diagonal(D, np.sum(A, axis=1))
    L = D - A
    return La, L
    #return L, D, A, np.mean(far_dist)


## Module for part 2 of homework
def consensus(world_size=50, num_agent=10, x_dimn=2, t=1, a_radius=10, r_radius=5, graphics=True, mode='dist'):
    X = random.uniform(-(world_size-1), (world_size-1), (num_agent, x_dimn))
    X = np.round(X)

    # List to store the eigen values
    lr_eigen_list = []
    la_eigen_list = []

    #### Change the integer for the experiments
    ## Change over here
    attraction_radius = a_radius
    attraction_knn = a_radius
    repulsion_radius = r_radius

    ## No matter what do repulsion by distance
    # Create Lapacian for repulsion
    # La, Lr = create_lalr(X, attraction_radius, repulsion_radius)

    if mode == 'dist':
        # Create Lapacian for attraction with radius 30
        # La, D, A = create_adjancey_graph(X, attraction_radius)
        La, Lr = create_lalr(X, attraction_radius, repulsion_radius)
    else:
        #La, D, A, _ = create_adjancey_graph_knn(X, attraction_knn)
        La, Lr = create_lalr_knn(X, attraction_radius, repulsion_radius)

    i = 0

    plt.ion()
    for dt in np.arange(0, t, 0.02):
        lr_eigen_list.append(np.sort(np.linalg.eig(Lr)[0])[1])
        la_eigen_list.append(np.sort(np.linalg.eig(La)[0])[1])
        dxdt = X[:,0]
        dydt = X[:,1]

        if graphics == True:
            plt.gca().cla() # optionally clear axes
            #plt.plot(dxdt, dydt)
            plt.scatter(dxdt, dydt, s=5, alpha=0.75, color='r')
            plt.title(i)
            plt.draw()
            plt.pause(0.02)

        dxdt = np.reshape(dxdt, (dxdt.shape[0], 1))
        dydt = np.reshape(dydt, (dydt.shape[0], 1))
        dxdt = np.dot((np.eye(X.shape[0]) + (Lr - La) * dt ), dxdt) + random.normal(0, 0.1,(dxdt.shape[0], 1)) * dt
        dydt = np.dot((np.eye(X.shape[0]) + (Lr - La) * dt ), dydt) + random.normal(0, 0.1,(dydt.shape[0], 1)) * dt
        X = np.hstack((dxdt, dydt))
        # Compute the La and Lr matrix again
        # Lr, D, A = create_adjancey_graph(X, 10)

        if mode == 'dist':
            La, Lr = create_lalr(X, attraction_radius, repulsion_radius)
        else:
            La, Lr = create_lalr_knn(X, attraction_radius, repulsion_radius)
        # print (i, X)
        i += 1

    plt.show(block=True)

    # Plot the eigen values
    draw_eigen_values(np.arange(0, t, 0.02), la_eigen_list, lr_eigen_list)

def draw_network(A):
    G = nx.from_numpy_matrix(A)
    nx.draw(G, with_labels=True)
    plt.show()

def run_randomnode(world_size=50, num_agent=10, x_dimn=2, time=1, radius=10, show_fig=False):
    X = random.uniform(-(world_size-1), (world_size-1), (num_agent, x_dimn))
    X = np.round(X)
    if show_fig:
        draw_agents(X, radius)
    L, D, A = create_adjancey_graph(X, radius)
    # print ('DA', D, A)
    # D.diagonal >=1
    return L, D, A

def run_randomnode_knn(world_size=50, num_agent=10, x_dimn=2, time=1, N=3, show_fig=False):
    X = random.uniform(-(world_size-1), (world_size-1), (num_agent, x_dimn))
    # X = np.array([[5,45], [10, 45], [15, 45], [5, 5], [10, 5], [15, 5]])
    X = np.round(X)
    if show_fig:
        draw_agents(X, N)
    L, D, A, fd = create_adjancey_graph_knn(X, N)
    return L, D, A, fd

def run_connectedness_prob(radius=5, mode='dist'):
    # Mode can me dist or knn
    if mode =='dist':
        L, D, A = run_randomnode(world_size=50, num_agent=20, radius=radius)
    elif mode == 'knn':
        L, D, A, ld = run_randomnode_knn(world_size=50, num_agent=20, N=radius)
    else:
        pass
    ## This logic is not correct
    # connected = D.diagonal() >= 1
    # connected = connected * 1
    # if connected.sum() == D.diagonal().shape[0]:
    #    return True
    ## compute eigen values. If filder eigen value is zero than the graph is disconnected
    ret = None
    if 0.0 == np.abs(np.sort(np.linalg.eig(L)[0])[1]):
        ret = False
    else:
        ret = True
    if mode == 'dist':
        return ret
    else:
        return ret, ld

def run_connectedness_expriment(num_exp=20, mode='dist'):
    connect_prob = []
    x = []
    lddist = []
    if mode == 'dist':
        iteration = range(5, 50, 5)
    else:
        iteration = range(2, 18, 1)

    for r in iteration:
        true_cnt = 0
        ld_list = []
        for i in range(num_exp):
            if mode == 'dist':
                ret = run_connectedness_prob(r, mode)
            else:
                ret, ld = run_connectedness_prob(r, mode)
                ld_list.append(ld)

            if ret == True:
                true_cnt += 1

        connect_prob.append((true_cnt*1.0)/num_exp)
        if mode == 'dist':
            x.append(np.round(r/50.0, 2))
        else:
            x.append(r)
            lddist.append(np.mean(np.array(ld_list)))

    draw_graph(x, connect_prob, mode=mode)
    if mode == 'knn':
        draw_distance(x, lddist)

#### Part 2


if __name__ == '__main__':
    # Run the connectedness experiment
    ## Part 1
    # mode in 'dist' or 'knn'
    # run_connectedness_expriment(200, mode='knn')

    ## Part 2
    ## Change parameters inside this function to answer the questions in the homework
    consensus(world_size=50, num_agent=25, x_dimn=2, t=1, a_radius=20, r_radius=2, graphics=True, mode='knn')
    # input("Push something will ya?")