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


def main():
    A = input()
    A = create_matrix(A)

    B = input()
    B = create_matrix(B)
    
    pi = input()
    pi = create_matrix(pi)
    
    emission = input()
    nb_emission, seq_emission = create_sequence(emission)
    
    alpha_list = [[0 for j in range(len(pi[0]))] for i in range(nb_emission)]
    for i in range(nb_emission):
        for j in range(len(pi[0])):
            if i == 0:
                alpha_list[i][j] = pi[0][j] * B[j][int(seq_emission[i])]
            else:
                for k in range(len(pi[0])):
                    alpha_list[i][j] += alpha_list[i-1][k] * A[k][j] * B[j][int(seq_emission[i])]
    proba = sum(alpha_list[-1])
    print(round(proba, 6))                
    

if __name__ == "__main__":
    main()
