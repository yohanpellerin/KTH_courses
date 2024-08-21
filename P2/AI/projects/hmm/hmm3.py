import math

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


def alpha_pass(A, B, pi, seq_emission):
    nb_emission = len(seq_emission)
    alpha_list = [[0 for j in range(len(pi[0]))] for i in range(nb_emission)]
    scale_factors = [0 for i in range(nb_emission)]
    for j in range(len(pi[0])):
        alpha_list[0][j] = pi[0][j] * B[j][int(seq_emission[0])]
        scale_factors[0] += alpha_list[0][j]
    scale_factors[0] = 1 / scale_factors[0]
    
    for j in range(len(pi[0])):
        alpha_list[0][j] = alpha_list[0][j] * scale_factors[0]
    
    for i in range(1, nb_emission):
        for j in range(len(pi[0])):
            for k in range(len(pi[0])):
                alpha_list[i][j] += alpha_list[i-1][k] * A[k][j] * B[j][int(seq_emission[i])]
            scale_factors[i] += alpha_list[i][j]
        scale_factors[i] = 1 / scale_factors[i]
        for j in range(len(pi[0])):
            alpha_list[i][j] = alpha_list[i][j] * scale_factors[i]
    return alpha_list, scale_factors


def beta_pass(A, B, pi, seq_emission, scale_factors):
    nb_emission = len(seq_emission)
    beta_list = [[0 for j in range(len(pi[0]))] for i in range(nb_emission)]
    for i in range(nb_emission-1, -1, -1):
        for j in range(len(pi[0])):
            if i == nb_emission - 1:
                beta_list[i][j] = 1 * scale_factors[i]
            else:
                for k in range(len(pi[0])):
                    beta_list[i][j] += A[j][k] * B[k][int(seq_emission[i+1])] * beta_list[i+1][k] * scale_factors[i]
    return beta_list


def di_gamma(A, B, pi, seq_emission, alpha_list, beta_list):
    nb_emission = len(seq_emission)
    di_gamma_list = [[[0 for k in range(len(pi[0]))] for j in range(len(pi[0]))] for i in range(nb_emission-1)]
    gamma_list = [[0 for j in range(len(di_gamma_list[0]))] for i in range(len(di_gamma_list))]
    for i in range(nb_emission-1):
        for j in range(len(pi[0])):
            for k in range(len(pi[0])):
                di_gamma_list[i][j][k] = alpha_list[i][j] * A[j][k] * B[k][int(seq_emission[i+1])] * beta_list[i+1][k]
                gamma_list[i][j] += di_gamma_list[i][j][k]
    return gamma_list, di_gamma_list


def baum_welch_update(A, B, pi, seq_emission, di_gamma_list, gamma_list):
    for i in range(len(pi[0])):
        den = 0
        for t in range(len(seq_emission)-1):
            den += gamma_list[t][i]
        for j in range(len(pi[0])):
            num = 0
            for t in range(len(seq_emission)-1):
                num += di_gamma_list[t][i][j]
            A[i][j] = num / den
    
    for i in range(len(pi[0])):
        den = 0
        for t in range(len(seq_emission)-1):
            den += gamma_list[t][i]
        for j in range(len(B[0])):
            num = 0
            for t in range(len(seq_emission)-1):
                if int(seq_emission[t]) == j:
                    num += gamma_list[t][i]
            B[i][j] = num / den
    
    for i in range(len(pi[0])):
        pi[0][i] = gamma_list[0][i]
    
    return A, B, pi
    

def baum_welch(A, B, pi, seq_emission):
    nb_emission = len(seq_emission)
    max_iters = 5000
    iter = 0
    not_converged = True
    prev_log_likelihood = float('-inf')
    
    while not_converged:
        alpha_list, scale_factors = alpha_pass(A, B, pi, seq_emission)
        beta_list = beta_pass(A, B, pi, seq_emission, scale_factors)
        gamma_list, di_gamma_list = di_gamma(A, B, pi, seq_emission, alpha_list, beta_list)
        new_A, new_B, new_pi = baum_welch_update(A, B, pi, seq_emission, di_gamma_list, gamma_list)
        
        iter += 1
        
        if iter == max_iters:
            not_converged = False
        else:
            A = new_A
            B = new_B
            pi = new_pi
            
            
            log_likelihood = -sum(math.log(scale) for scale in scale_factors)

            if log_likelihood <= prev_log_likelihood:
                not_converged = False
            prev_log_likelihood = log_likelihood
            
    print(f"Converged after {iter} iterations")
    return A, B


def create_string(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    string = str(rows) + ' ' + str(cols)
    for i in range(rows):
        for j in range(cols):
            string += ' ' + str(round(matrix[i][j],6))
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
    
    A, B = baum_welch(A, B, pi, seq_emission)
    
    print(create_string(A))
    print(create_string(B))                
    

if __name__ == "__main__":
    main()
    
