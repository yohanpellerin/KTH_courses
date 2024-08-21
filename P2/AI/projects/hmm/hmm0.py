def create_matrix(string):
    list_str = string.split()
    list_val = [float(i) for i in list_str]
    rows = int(list_val[0])
    cols = int(list_val[1])
    list_val = list_val[2:]
    matrix = [list_val[i*cols:(i+1)*cols] for i in range(rows)]
    return matrix


def multiply(A, B):
    if len(A[0]) != len(B):
        return [[0]]
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            value = 0
            for k in range(len(B)):
                value += A[i][k] * B[k][j]
            row.append(value)
        result.append(row)
    return result


def create_string(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    string = str(rows) + ' ' + str(cols)
    for i in range(rows):
        for j in range(cols):
            string += ' ' + str(matrix[i][j])
    return string


def main():
    A = input()
    A = create_matrix(A)

    B = input()
    B = create_matrix(B)

    current_state_dis = input()
    current_state_dis = create_matrix(current_state_dis)

    next_observartion_dis = multiply(multiply(current_state_dis, A), B)
    print(create_string(next_observartion_dis))


if __name__ == "__main__":
    main()
