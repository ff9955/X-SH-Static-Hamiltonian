import numpy as np
import matplotlib.pyplot as plt

def build_lattice(x_molecules, y_molecules, lattice_spacing):
    '''
    function builds a 1D/2D lattice with a prescribed number of evenly spaced molecules
    along the x/y-axes

    input: number of molecules along x and y-axes; lattice_spacing = distant (in angstroms)
    between adjacent molecules

    output: 2D array with two columns and n rows, each row contains x/y-coords
    of a single particle
    '''

    coordinate_list = []

    for index in range(x_molecules):
        for index2 in range(y_molecules):

            #nested loop that iterates over number of x and y-molecules
            #single x-mol taken, and iterated over all molecules over y-axes

            coordinate_list.append([index, index2])
            #appending list of x/y indeces of each particle, e.g [0,0] corresponds 
            #first y-molecule of first molecule along x-axis

    coordinate_vector = np.array(coordinate_list) #converting list of lists to 2D array
    coordinate_vector = coordinate_vector*lattice_spacing

    #distant between adjacent points is constant, so coordinates of each molecule are 
    #equal to indeces multiplied by spacing, e.g [0,1] = [0,5] as 0th x-molecules has
    #no distance, whereas 1st y-molecule is 5A away from 0th y-molecule
    

    return coordinate_vector


def DA_index(coord_vector, donors):
    '''
    function that sets indeces of donor/acceptor phase molecules using their coordinates

    input: 2D array of each molecules's coordinates, number of donors you want in your system

    output: 2 lists: one of the donor indeces and one with the acceptor indeces
    '''

    total_molecules = len(coord_vector)

    donor_indeces = [index for index in range(total_molecules)[:donors]]
    #donor indeces assinged by just taking the first 'donors' molecules in the lattice
    acceptor_indeces = [index2 for index2 in range(total_molecules)[donors:]]
    #acceptor indeces obtained by taking the remainder

    return donor_indeces, acceptor_indeces


def build_connectivity(coord_vector, max_distance):
    '''
    function that tells which molecules are nearest neighbours

    input: vector of lattice coordinates, distance cutoff beyond which points 
    are not nearest neighbours

    output: list of lists, where each list contains indeces of two molecules that 
    are closer than the distance cutoff
    '''

    connectivity_list = []
    
    for index in range(len(coord_vector)):
        #looping over all coordinates in coord_vector by indexing the array with 'index'

        if index == (len(coord_vector) - 1): break
        #breaks the for loop when you've reached the last element in range; avoids indexing error

        for index2 in range(index + 1, len(coord_vector)):
            #looping over the molecules that come after the molecule selected by 'index'

            euclidean_distance = np.sqrt((np.sum((coord_vector[index] - coord_vector[index2])**2)))
            #calculating distance between two molecules obtained from coord vector via 
            #'index' and 'index2' 

            if euclidean_distance <= max_distance:

                connectivity_list.append([index, index2])
                #if the two selected molecules are closer than the cutoff distance, a 2-component list
                #if appended to 'connectivity_list', where the components are the indeces of the
                #molecules in 'coord_vector'


    return connectivity_list

#it is important to understand that the indeces describing the molecules in the donor/acceptor
#lists and connectivity lists are the same


def define_CT_states(donor_list, acceptor_list):
    '''
    function that defines all possible charge transfer basis states of the system

    input: lists of indeces of molecules in donor and acceptor phases

    output: list of lists, where each list contains one donor and one acceptor index
    '''

    state_list = []

    for element in donor_list:
        for element2 in acceptor_list:
            state_list.append([element, element2])
    
    return state_list


def define_XT_states(molecule_list):
    '''
    function that defines all possible exciton basis states of the system

    input: lists of indeces of molecules in donor or acceptor phase

    output: list of lists, where each list contains the index of a molecule twice,
    since both charge carriers are on the same molecule in a Frenkel exciton
    '''

    state_list = []

    for element in molecule_list:

        state_list.append([element, element])
    
    return state_list


def build_CT_block(state_list, coord_vector, connectivity, interaction_constant, coupling_constant):
    '''
    this function builds the CT block (site energies and couplings) of the electronic Hamiltonian

    input: list of CT-states, list of nearest neighbours, interaction constant of coulombic
    interaction between charge carriers, coupling magnitude between charge carriers

    output: NxN tight binding Hamiltonian matrix, where N = number of CT-states
    '''

    number_states = len(state_list)
    H_matrix = np.zeros((number_states, number_states))

    for number in range(number_states):
        #looping over all states in state list

        for index in range(number_states)[number:]:
            #ignoring the states that come before 'index' in the states list

            if state_list[number] == state_list[index]:
                #if the states are the same, assign a site energy

                donor_molecule = state_list[number][0]
                acceptor_molecule = state_list[number][1]
                #obtain the indeces of the donor and acceptor molecules

                euclidean_distance = np.sqrt((np.sum((coord_vector[donor_molecule] - coord_vector[acceptor_molecule])**2)))
                single_lattice_distance = np.sum(coord_vector[1] - coord_vector[0])
                #calculate distance between electron and hole

                euclidean_distance = euclidean_distance/single_lattice_distance
                #express this distance in units of lattice spacing

                H_matrix[number,index] = (interaction_constant)/euclidean_distance
                #assign the site energy, which is changes inversely with the distance
                H_matrix[number, index] = H_matrix[number, index]/2
                #divide by two as we're only assigning the top matrix triangle

            elif (0 in np.subtract(state_list[number], state_list[index])):
                # if the states are different, check that they are nearest neighbours before assigning an electronic coupling
                
                for element in connectivity:
                    if (state_list[number][0] == element[0]) and (state_list[index][0] == element[1]):
                        #this checks if the donor molecules are nearest neighbours
                        H_matrix[number, index] = coupling_constant
                        break

                    elif (state_list[number][1] == element[0]) and (state_list[index][1] == element[1]):
                        #this checks if the acceptor molecules are nearest neighbours
                        H_matrix[number, index] = coupling_constant
                        break
    
    H_matrix = H_matrix + H_matrix.T
    #assigning the bottom triangle with transpose of the top triangle, as the matrix is symmetric

    return H_matrix


def build_XT_block(state_list, connectivity, interaction_constant, coupling_constant):
    '''
    this function builds the XT block (site energies and couplings) of the electronic Hamiltonian

    input: list of XT-states, list of nearest neighbours, interaction constant of coulombic
    interaction between charge carriers, coupling magnitude between charge carriers

    output: MxM tight binding Hamiltonian matrix, where M = number of XT-states
    '''

    number_states = len(state_list)
    H_matrix = np.zeros((number_states, number_states))
    
    for number in range(number_states):
        #looping over all states in state list

        for index in range(number_states)[number:]:
            #ignoring the states that come before 'index' in the states list

            if state_list[number] == state_list[index]: 
               H_matrix[number, index] = interaction_constant/2
            #if the states are the same, assign a site energy
            
            else:
                for element in connectivity:
                    
                    # if the states are different, check that they are nearest neighbours before assigning an electronic coupling

                    if (state_list[number][0] in element) and (state_list[index][0] in element):

                        H_matrix[number, index] = coupling_constant
                        break

    H_matrix = H_matrix + H_matrix.T

    return H_matrix


def build_XT_CT_block(XT_states, CT_states, connectivity, coupling_constant):
    '''
    function that assigns the offdiagonal block of the full electronic Hamiltonian
    matrix, which describes coupling between XT and CT states

    input: list of CT/XT states, list of nearest neighbours, coupling constant

    output: NxM matrix, where N = number of CT-states, and M = number of XT states
    '''

    CT_index = 0
    number_XT_states = len(XT_states)
    number_CT_states = len(CT_states)

    H_matrix = np.zeros((number_CT_states, number_XT_states))

    if (XT_states[0][0] == CT_states[0][0]):
        #this checks if the excitons are in the donor or acceptor phase
        CT_index = 1
    else:
        CT_index = 0

    #reminder: donor molecules are first in the state list, acceptors are second

    for number in range(number_CT_states):

        for index in range(number_XT_states):

            if (0 in np.subtract(CT_states[number], XT_states[index])):
                #this checks that the non-nearest neighbour indeces are equal, since
                # coupling between charge carriers on four different sites is zero

                for element in connectivity:
                    if (CT_states[number][CT_index] in element) and (XT_states[index][CT_index] in element):
                        #if either electrons or holes are on nearest neighbour sites, coupling is assigned

                        H_matrix[number, index] = coupling_constant
    
    return H_matrix


def build_full_Hamiltonian(CT_block, XT_block, XT_CT_block):
    '''
    function that combines CT, XT, and CT-XT blocks to create full state-space 
    Hamiltonian matrix where the CT and XT blocks are ordered in the same way each time

    input: different matrix blocks as defined above

    output: DxD matrix where D = total number of states
    '''

    number_CT_states = len(CT_block)
    number_XT_states = len(XT_block)
    total_states = number_CT_states + number_XT_states

    H_matrix = np.zeros((total_states, total_states))

    H_matrix[0:number_CT_states, 0:number_CT_states] = CT_block
    H_matrix[number_CT_states:total_states, number_CT_states:total_states] = XT_block
    #assigning the two diagonal blocks
    
    H_matrix[0:number_CT_states, number_CT_states:total_states] = XT_CT_block
    #assign the top-right XT-CT block

    H_matrix[number_CT_states: total_states, 0:number_CT_states] = XT_CT_block.T
    #assign the bottom-left XT-CT block by taking the transpose of the one above

    return H_matrix


def get_eigen(matrix):
    '''
    function computes energies and eigenvectors of eigenstates in increasing order of energy

    input: square Hamiltonian matrix

    output: row vector (length D) of eigen energies, matrix (DxD) of orthonormal eigenvectors
    where each column is a single eigenvector
    '''
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    idx = eigenvalues.argsort()
    #print idx
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors
#    return eigenvalues, eigenvectors[0, :], eigenvectors[:, 0], eigenvectors[:, 1]


def CT_site_populations(eigenvector, donors, acceptors):
    '''
    function that returns the distribution of a single charge carrier's eigenfunction
    across the system's lattice sites

    input: single eigenvector of Hamiltonian, lists of donor/acceptor indeces

    output: list of length equal to number of lattice sites; each element corresponds to
    the proportion of the eigenfunction that is present on this lattice site
    '''

    number_CT_states = len(donors)*len(acceptors)
    CT_states_per_donor = int(number_CT_states/len(donors))
    #each donor contributes the same number of CT-states to the overall number of CT-states

    CT_eigenvector = eigenvector[:number_CT_states]
    #remove the elements of the total eigenvector that correspond to the expansion coefficients
    #of the excitonic basis states
    donor_site_populations = []

    for index in range(0, number_CT_states, CT_states_per_donor):
        #iterate over list of CT-states, taking slices at a time, where each slice contains all
        #possible CT-states of a single donor molecule

        cumulative_population = np.sum(CT_eigenvector[index: index + CT_states_per_donor]**2)
        #we sum the populations of these CT-states, this tells us the total population that is present
        #on this one donor molecule
        donor_site_populations.append(cumulative_population)

    acceptor_site_populations = []
    cumulative_population = []
    #same principle holds for acceptor, although we index differently, as the state lists
    #were made by summing over all acceptor molecules per donor

    for acceptor_number in range(0, len(acceptors)):

        single_acceptor_populations = [CT_eigenvector[number2]**2 for number2 in range(0+acceptor_number, number_CT_states, CT_states_per_donor)]
        #list comprehension gets all CT-states corresponding to a single donor by skipping N elements at a time, where N = number of CT-states per donor
        #each donor has a CT-state with just one acceptor, so an acceptor's adjacent CT-states are always separated by N list elements

        cumulative_population = np.sum(single_acceptor_populations)
        #sum populations of all CT-states of a single acceptor

        acceptor_site_populations.append(cumulative_population)

    return donor_site_populations, acceptor_site_populations 


def XT_site_populations(eigenvector, XT_states):
    '''
    function which calculates how the excitonic populations of a charge carrier's eigenfunction is distributed over 
    a lattice

    input: eigenvector of a Hamiltonian, list of excitonic basis states

    output: list where each element shows the proportion of the eigenfunction that is present on a lattice
    site as an exciton
    '''

    number_XT_states = len(XT_states)
    XT_eigenvector = eigenvector[-number_XT_states:]

    site_populations = []

    for coefficient in XT_eigenvector:
        site_populations.append(coefficient**2)
    
    #simply summing over the squared elements of the eigenvector's XT-section here, since
    #each lattice site has just one possible exciton

    return site_populations


def integrated_CT_populations(donor_site_populations, acceptor_site_populations, y_molecules):
    '''
    function that sums the lattice sites' relative populations along the y-axis to show how
    an eigenfunction's CT-component is distributed over columns of the lattice, not only
    individual sites

    input: list of CT-state populations of donor and acceptor sites; number of molecules per 
    column in the lattice (y-axis)

    output: lists with length equal to half the number of molecules along the x-axis, each
    element corresponds to the proportion of the eigenfunction's CT-component on a certain column 
    '''

    integrated_donor_populations = [np.sum(donor_site_populations[number:number+y_molecules]) for number in range(0,len(donor_site_populations),y_molecules)]

    #these lists comprehensions just iterate over the per-site population lists, and sum over every N molecules, where N = the number of molecules in a single column

    integrated_acceptor_populations = [np.sum(acceptor_site_populations[number2:number2+y_molecules]) for number2 in range(0,len(acceptor_site_populations),y_molecules)]

    return integrated_donor_populations, integrated_acceptor_populations


def integrated_XT_populations(XT_populations, y_molecules):
    '''
    function that sums over relative XT-populations for each column of the lattice - same as above

    input: list of relative presence of eigenfunction's XT-component per site, number of molecules
    in a single column of the lattice

    output: lists with length equal to half the number of molecules along the x-axis, each
    element corresponds to the proportion of the eigenfunction's XT-component on a certain column 
    '''

    integrated_populations = [np.sum([XT_populations[number:number+y_molecules]]) for number in range(0, len(XT_populations), y_molecules)]

    return integrated_populations