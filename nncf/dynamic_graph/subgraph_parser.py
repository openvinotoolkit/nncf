"""
 Copyright (c) 2019-2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import nncf.dynamic_graph.patterns as p

class TokenType:
    VAR, ADD, OR, LP, RP= ["Var", "+", "|", "(", ")"]#range(6)

class TreeNode:
    tokenType = None
    value = None
    l = None
    r = None

    def __init__(self, tokenType, l = None, r = None):
        self.tokenType = tokenType
        self.l = l
        self.r = r

class AddNode(TreeNode) :
    def __init__(self, l, r):
        super().__init__(TokenType.ADD, l,r)
        self.value = l.value  + r.value 

class OrNode(TreeNode) :
    def __init__(self, l, r):
        super().__init__(TokenType.OR, l,r)
        self.value = l.value | r.value

class Tokenizer:
    tokens = None
    tokenTypes = None
    exp = None
    idx = 0

    def __init__(self, exp):
        self.expression = exp

    def next(self):
        self.idx += 1
        return self.tokens[self.idx-1]

    def peek(self):
        return self.tokens[self.idx]

    def has_next(self):
        return self.idx < len(self.tokens)

    def next_token_type(self):
        return self.tokenTypes[self.idx]

    def tokenize(self):
        import re
        reg = re.compile(r'(\+|ADD|\||OR|\(|\))') 
        self.tokens = reg.split(self.expression)
        self.tokens = [t.strip() for t in self.tokens if t.strip() != '']

        self.tokenTypes = []
        for t in self.tokens:
            if t == '+' or t == '&' or t == 'ADD':
                self.tokenTypes.append(TokenType.ADD)
            elif t == '|' or t == 'OR':
                self.tokenTypes.append(TokenType.OR)
            elif t == '(':
                self.tokenTypes.append(TokenType.LP)
            elif t == ')':
                self.tokenTypes.append(TokenType.RP)
            else :
                self.tokenTypes.append(TokenType.VAR)

class SubgraphParser:
    tokenizer = None
    root = None

    def __init__(self):
        self.macro_dict = p.pattern_dict

    def parse(self, exp):
        self.tokenizer = Tokenizer(exp)
        self.tokenizer.tokenize()
        self.root = self.parse_expression()
        return self.root.value

    def parse_expression(self):
        term = self.parse_add()
        while self.tokenizer.has_next() and self.tokenizer.next_token_type() == TokenType.OR:
            #print(" OR ", end = '')
            self.tokenizer.next()
            right = self.parse_add()
            term = OrNode(term,right)
        return term

    def parse_add(self):
        term = self.parse_sub_expr()
        while self.tokenizer.has_next() and self.tokenizer.next_token_type() == TokenType.ADD:
            #print(" + ", end = '')
            self.tokenizer.next()
            right = self.parse_sub_expr()
            term = AddNode(term, right)
        return term

    def parse_sub_expr(self):
        if self.tokenizer.has_next() and self.tokenizer.next_token_type() == TokenType.LP:
            #print("(")
            self.tokenizer.next()
            expression = self.parse_expression()
            if self.tokenizer.has_next() and self.tokenizer.next_token_type() == TokenType.RP:
                #print(")")
                self.tokenizer.next()
                return expression
            else:
                raise Exception("Closing ) expected, but got " + self.tokenizer.next())

        leaf1 = self.parse_leaf()
        return leaf1

    def eval_token(self, token_value):
        # print(token_value)
        if not token_value in self.macro_dict:
            raise Exception("Unknown symbol " + token_value + " encountered in graph expression")        
        return self.macro_dict[token_value]

    def parse_leaf(self):
        if self.tokenizer.has_next():
            tokenType = self.tokenizer.next_token_type()
            if tokenType == TokenType.VAR:
                n = TreeNode(tokenType)
                n.value = self.eval_token(self.tokenizer.next())
                return n
            else:
                raise Exception('VAR expected, but got ' + self.tokenizer.next())


