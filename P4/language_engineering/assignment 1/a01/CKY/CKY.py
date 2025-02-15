from terminaltables import AsciiTable
import argparse
import numpy as np

"""
The CKY parsing algorithm.

This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2019 by Johan Boye.
"""

class CKY :

    # The unary rules as a dictionary from words to non-terminals,
    # e.g. { cuts : [Noun, Verb] }
    unary_rules = {}

    # The binary rules as a dictionary of dictionaries. A rule
    # S->NP,VP would result in the structure:
    # { NP : {VP : [S]}} 
    binary_rules = {}

    # The parsing table
    table = []

    # The backpointers in the parsing table
    backptr = []

    # The words of the input sentence
    words = []


    # Reads the grammar file and initializes the 'unary_rules' and
    # 'binary_rules' dictionaries
    def __init__(self, grammar_file) :
        stream = open( grammar_file, mode='r', encoding='utf8' )
        for line in stream :
            rule = line.split("->")
            left = rule[0].strip()
            right = rule[1].split(',')
            if len(right) == 2 :
                # A binary rule
                first = right[0].strip()
                second = right[1].strip()
                if first in self.binary_rules :
                    first_rules = self.binary_rules[first]
                else :
                    first_rules = {}
                    self.binary_rules[first] = first_rules
                if second in first_rules :
                    second_rules = first_rules[second]
                    if left not in second_rules :
                        second_rules.append( left )
                else :
                    second_rules = [left]
                    first_rules[second] = second_rules
            if len(right) == 1 :
                # A unary rule
                word = right[0].strip()
                if word in self.unary_rules :
                    word_rules = self.unary_rules[word]
                    if left not in word_rules :
                        word_rules.append( left )
                else :
                    word_rules = [left]
                    self.unary_rules[word] = word_rules


    # Parses the sentence a and computes all the cells in the
    # parse table, and all the backpointers in the table
    def parse(self, s) :
        self.words = s.split()        
        #  YOUR CODE HERE
        for i in range(len(self.words)):
            self.table.append([])
            self.backptr.append([])
            for j in range(len(self.words)-1,-1,-1):
                if j<i:
                    self.table[i]=[[]]+self.table[i]
                    self.backptr[i]=[[]]+self.backptr[i]
                    for k in range (j,i):
                        for x in self.table[k][j]:
                            for y in self.table[i][k-j+1]:
                                if x in self.binary_rules and y in self.binary_rules[x]:
                                    for z in self.binary_rules[x][y]:
                                        self.table[i][0]=self.table[i][0]+[z]
                                        self.backptr[i][0]=self.backptr[i][0]+[([x,y],(j,k,k+1,i))]
                                    
                if j==i:
                    #add unary rules
                    self.table[i]=[self.unary_rules[self.words[i]]]+self.table[i]
                    self.backptr[i]= [None]+self.backptr[i]

                elif j>i:    
                    self.table[i].append([])
                    self.backptr[i].append([])
        #invert the table
        self.table=np.array(self.table, dtype=object).T
        self.table=self.table.tolist()
        self.backptr=np.array(self.backptr, dtype=object).T
        self.backptr=self.backptr.tolist()


    # Prints the parse table
    def print_table( self ) :
        t = AsciiTable(self.table)
        t.inner_heading_row_border = False
        print( t.table )


    # Prints all parse trees derivable from cell in row 'row' and
    # column 'column', rooted with the symbol 'symbol'
    def print_trees(self, column,row, symbol,reccursif=False) :
        #
        #  YOUR CODE HERE
        #
        sentence=[]
        if not(reccursif) and symbol not in self.table[row][column]:
            return ("No parse tree")
        else:
            #indices of the symbols in the cell
            indices=[i for i, x in enumerate(self.table[row][column]) if x == symbol]
            #for each symbol in the cell
            for i in indices:
                if row==column:
                    return [symbol+"("+self.words[row]+")"]
                else:
                    #get the backpointer
                    back=self.backptr[row][column][i]
                    #print the left tree
                    left_tree = self.print_trees(back[1][1], back[1][0], back[0][0],True)
                #print the right tree
                    right_tree = self.print_trees(back[1][3], back[1][2], back[0][1],True)
                    
                    for left in left_tree:
                        for right in right_tree:
                            sentence.append(symbol+"("+left+","+right+")")
                #print the symbol
        return sentence
                


def main() :

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CKY parser')
    parser.add_argument('--grammar', '-g', type=str,  required=True, help='The grammar describing legal sentences.')
    parser.add_argument('--input_sentence', '-i', type=str, required=True, help='The sentence to be parsed.')
    parser.add_argument('--print_parsetable', '-pp', action='store_true', help='Print parsetable')
    parser.add_argument('--print_trees', '-pt', action='store_true', help='Print trees')
    parser.add_argument('--symbol', '-s', type=str, default='S', help='Root symbol')

    arguments = parser.parse_args()

    cky = CKY( arguments.grammar )
    cky.parse( arguments.input_sentence )
    if arguments.print_parsetable :
        cky.print_table()
    if arguments.print_trees :
        cky.print_table()
        sentences=cky.print_trees( len(cky.words)-1, 0, arguments.symbol )
        for sentence in sentences:
            print(sentence)

    

if __name__ == '__main__' :
    main()    


                        
                        
                        
                    
                
                    

                
        
    
