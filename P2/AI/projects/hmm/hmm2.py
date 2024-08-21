def create_matrix(string):
    list_str = string.split()
    list_val = [float(i) for i in list_str]
    rows = int(list_val[0])
    cols = int(list_val[1])
    list_val = list_val[2:]
    matrix = [list_val[i*cols:(i+1)*cols] for i in range(rows)]
    return matrix


def create_sequence(string):
    list_str = string.split()
    list_val = [float(i) for i in list_str]
    if len(list_val[1:]) != list_val[0]:
        return 0, [0]
    return int(list_val[0]), list_val[1:]


def create_string(list):
    string = ''
    for i in range(len(list)):
        string += str(int(list[i]))
        if i != len(list) - 1:
            string += ' '
    return string


def main():
    A = input()
    A = create_matrix(A)

    B = input()
    B = create_matrix(B)
    
    pi = input()
    pi = create_matrix(pi)
    
    emission = input()
    nb_emission, seq_emission = create_sequence(emission)
    
    delta = [[0 for j in range(len(pi[0]))] for i in range(nb_emission)]
    delta_idx = [[0 for j in range(len(pi[0]))] for i in range(nb_emission)]
    for i in range(nb_emission):
        for j in range(len(pi[0])):
            if i == 0:
                delta[i][j] = pi[0][j] * B[j][int(seq_emission[i])]
            else:
                probabilities = []
                for k in range(len(pi[0])):
                    probabilities.append(delta[i - 1][k] * A[k][j] * B[j][int(seq_emission[i])])
                delta[i][j] = max(probabilities)
                delta_idx[i][j] = probabilities.index(max(probabilities))
    
    hidden_states = [0 for i in range(nb_emission)]
    hidden_states[-1] = delta[-1].index(max(delta[-1]))
    
    for i in range(nb_emission - 2, -1, -1):
        hidden_states[i] = delta_idx[i + 1][hidden_states[i + 1]]
        
    print(create_string(hidden_states))

if __name__ == "__main__":
    main()