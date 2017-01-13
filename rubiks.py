import numpy as np
from seispy.util import MultiThreadProcess

VERBOSE = False

class TwoByTwo:
    def __init__(self):
        self.state = np.empty(shape=(6, 2, 2), dtype=str)
        self.state[0, 0] = np.array(["W", "W"])
        self.state[0, 1] = np.array(["W", "W"])
        self.state[1, 0] = np.array(["O", "O"])
        self.state[1, 1] = np.array(["O", "O"])
        self.state[2, 0] = np.array(["Y", "Y"])
        self.state[2, 1] = np.array(["Y", "Y"])
        self.state[3, 0] = np.array(["R", "R"])
        self.state[3, 1] = np.array(["R", "R"])
        self.state[4, 0] = np.array(["B", "B"])
        self.state[4, 1] = np.array(["B", "B"])
        self.state[5, 0] = np.array(["G", "G"])
        self.state[5, 1] = np.array(["G", "G"])
        self.sequence = []
        self.transformations = {1: self.U,
                                2: self.Uprime,
                                3: self.D,
                                4: self.Dprime,
                                5: self.R,
                                6: self.Rprime,
                                7: self.L,
                                8: self.Lprime,
                                9: self.F,
                                10: self.Fprime,
                                11: self.B,
                                12: self.Bprime}

    def __eq__(self, other):
        for boolean in self.state.flat == other.state.flat:
            if not boolean:
                return False
        return True

    def __str__(self):
        s = "     -+-\n"
        s += "    |%s|%s|\n" % (self.state[2, 0, 0], self.state[2, 0, 1])
        s += "     -+-\n"
        s += "    |%s|%s|\n" % (self.state[2, 1, 0], self.state[2, 1, 1])
        s += "     -+-\n"
        s += "    |%s|%s|\n" % (self.state[3, 0, 0], self.state[3, 0, 1])
        s += "     -+-\n"
        s += "    |%s|%s|\n" % (self.state[3, 1, 0], self.state[3, 1, 1])
        s += " -+-+-+-+-+-\n"
        s += "|%s|%s|" % (self.state[4, 0, 0], self.state[4, 0, 1])
        s += "%s|%s|" % (self.state[0, 0, 0], self.state[0, 0, 1])
        s += "%s|%s|\n" % (self.state[5, 0, 0], self.state[5, 0, 1])
        s += " -+-+-+-+-+-\n"
        s += "|%s|%s|" % (self.state[4, 1, 0], self.state[4, 1, 1])
        s += "%s|%s|" % (self.state[0, 1, 0], self.state[0, 1, 1])
        s += "%s|%s|\n" % (self.state[5, 1, 0], self.state[5, 1, 1])
        s += " -+-+-+-+-+-\n"
        s += "    |%s|%s|\n" % (self.state[1, 0, 0], self.state[1, 0, 1])
        s += "     -+-\n"
        s += "    |%s|%s|\n" % (self.state[1, 1, 0], self.state[1, 1, 1])
        s += "     -+-\n"
        return s

    def print_transformation(self, new_state):
        s = "     -+-               -+-\n"
        s += "    |%s|%s|             |%s|%s|\n" % (self.state[2, 0, 0],
                                                 self.state[2, 0, 1],
                                                 new_state[2, 0, 0],
                                                 new_state[2, 0, 1])
        s += "     -+-               -+-\n"
        s += "    |%s|%s|             |%s|%s|\n" % (self.state[2, 1, 0],
                                                 self.state[2, 1, 1],
                                                 new_state[2, 1, 0],
                                                 new_state[2, 1, 1])
        s += "     -+-               -+-\n"
        s += "    |%s|%s|             |%s|%s|\n" % (self.state[3, 0, 0],
                                                 self.state[3, 0, 1],
                                                 new_state[3, 0, 0],
                                                 new_state[3, 0, 1])
        s += "     -+-               -+-\n"
        s += "    |%s|%s|             |%s|%s|\n" % (self.state[3, 1, 0],
                                                 self.state[3, 1, 1],
                                                 new_state[3, 1, 0],
                                                 new_state[3, 1, 1])
        s += " -+-+-+-+-+-       -+-+-+-+-+-\n"
        s += "|%s|%s|" % (self.state[4, 0, 0], self.state[4, 0, 1])
        s += "%s|%s|" % (self.state[0, 0, 0], self.state[0, 0, 1])
        s += "%s|%s|     " % (self.state[5, 0, 0], self.state[5, 0, 1])
        s += "|%s|%s|" % (new_state[4, 0, 0], new_state[4, 0, 1])
        s += "%s|%s|" % (new_state[0, 0, 0], new_state[0, 0, 1])
        s += "%s|%s|\n" % (new_state[5, 0, 0], new_state[5, 0, 1])
        s += " -+-+-+-+-+-  ==>  -+-+-+-+-+-\n"
        s += "|%s|%s|" % (self.state[4, 1, 0], self.state[4, 1, 1])
        s += "%s|%s|" % (self.state[0, 1, 0], self.state[0, 1, 1])
        s += "%s|%s|     " % (self.state[5, 1, 0], self.state[5, 1, 1])
        s += "|%s|%s|" % (new_state[4, 1, 0], new_state[4, 1, 1])
        s += "%s|%s|" % (new_state[0, 1, 0], new_state[0, 1, 1])
        s += "%s|%s|\n" % (new_state[5, 1, 0], new_state[5, 1, 1])
        s += " -+-+-+-+-+-       -+-+-+-+-+-\n"
        s += "    |%s|%s|             |%s|%s|\n" % (self.state[1, 0, 0],
                                                 self.state[1, 0, 1],
                                                 new_state[1, 0, 0],
                                                 new_state[1, 0, 1])
        s += "     -+-               -+-\n"
        s += "    |%s|%s|             |%s|%s|\n" % (self.state[1, 1, 0],
                                                 self.state[1, 1, 1],
                                                 new_state[1, 1, 0],
                                                 new_state[1, 1, 1])
        s += "     -+-               -+-\n"
        s += "==============================="
        print s

    def U(self, verbose=False):
        if VERBOSE or verbose:
            print "U"
        _state = np.copy(self.state)
        _state[0] = np.rot90(self.state[0], k=-1)
        _state[1, 0] = self.state[5, -1::-1, 0]
        _state[4, :, 1] = self.state[1, 0]
        _state[3, 1] = self.state[4, -1::-1, 1]
        _state[5, :, 0] = self.state[3, 1]
        if VERBOSE or verbose:
            self.print_transformation(_state)
        self.state = _state

    def Uprime(self, verbose=False):
        if VERBOSE or verbose:
            print "Uprime"
        _state = np.copy(self.state)
        _state[0] = np.rot90(self.state[0])
        _state[1, 0] = self.state[4, :, 1]
        _state[4, :, 1] = self.state[3, 1, -1::-1]
        _state[3, 1] = self.state[5, :, 0]
        _state[5, :, 0] = self.state[1, 0, -1::-1]
        if VERBOSE or verbose:
            self.print_transformation(_state)
        self.state = _state

    def D(self, verbose=False):
        if VERBOSE or verbose:
            print "D"
        _state = np.copy(self.state)
        _state[2] = np.rot90(self.state[2])
        _state[1, 1] = self.state[5, -1::-1, 1]
        _state[4, :, 0] = self.state[1, 1]
        _state[3, 0] = self.state[4, -1::-1, 0]
        _state[5, :, 1] = self.state[3, 0]
        if VERBOSE or verbose:
            self.print_transformation(_state)
        self.state = _state

    def Dprime(self, verbose=False):
        if VERBOSE or verbose:
            print "Dprime"
        _state = np.copy(self.state)
        _state[2] = np.rot90(self.state[2], k=-1)
        _state[1, 1] = self.state[4, :, 0]
        _state[4, :, 0] = self.state[3, 0, -1::-1]
        _state[3, 0] = self.state[5, :, 1]
        _state[5, :, 1] = self.state[1, 1, -1::-1]
        if VERBOSE or verbose:
            self.print_transformation(_state)
        self.state = _state

    def R(self, verbose=False):
        if VERBOSE or verbose:
            print "R"
        _state = np.copy(self.state)
        _state[5] = np.rot90(self.state[5])
        _state[0, :, 1] = self.state[3, :, 1]
        _state[1, :, 1] = self.state[0, :, 1]
        _state[2, :, 1] = self.state[1, :, 1]
        _state[3, :, 1] = self.state[2, :, 1]
        if VERBOSE or verbose:
            self.print_transformation(_state)
        self.state = _state

    def Rprime(self, verbose=False):
        if VERBOSE or verbose:
            print "Rprime"
        _state = np.copy(self.state)
        _state[5] = np.rot90(self.state[5], k=-1)
        _state[0, :, 1] = self.state[1, :, 1]
        _state[1, :, 1] = self.state[2, :, 1]
        _state[2, :, 1] = self.state[3, :, 1]
        _state[3, :, 1] = self.state[0, :, 1]
        if VERBOSE or verbose:
            self.print_transformation(_state)
        self.state = _state

    def L(self, verbose=False):
        if VERBOSE or verbose:
            print "L"
        _state = np.copy(self.state)
        _state[4] = np.rot90(self.state[4], k=-1)
        _state[0, :, 0] = self.state[3, :, 0]
        _state[1, :, 0] = self.state[0, :, 0]
        _state[2, :, 0] = self.state[1, :, 0]
        _state[3, :, 0] = self.state[2, :, 0]
        if VERBOSE or verbose:
            self.print_transformation(_state)
        self.state = _state

    def Lprime(self, verbose=False):
        if VERBOSE or verbose:
            print "Lprime"
        _state = np.copy(self.state)
        _state[4] = np.rot90(self.state[4])
        _state[0, :, 0] = self.state[1, :, 0]
        _state[1, :, 0] = self.state[2, :, 0]
        _state[2, :, 0] = self.state[3, :, 0]
        _state[3, :, 0] = self.state[0, :, 0]
        if VERBOSE or verbose:
            self.print_transformation(_state)
        self.state = _state

    def F(self, verbose=False):
        if VERBOSE or verbose:
            print "F"
        _state = np.copy(self.state)
        _state[1] = np.rot90(self.state[1], k=-1)
        _state[0, 1] = self.state[4, 1]
        _state[5, 1] = self.state[0, 1]
        _state[2, 0] = self.state[5, 1, -1::-1]
        _state[4, 1] = self.state[2, 0, -1::-1]
        if VERBOSE or verbose:
            self.print_transformation(_state)
        self.state = _state

    def Fprime(self, verbose=False):
        if VERBOSE or verbose:
            print "Fprime"
        _state = np.copy(self.state)
        _state[1] = np.rot90(self.state[1])
        _state[0, 1] = self.state[5, 1]
        _state[5, 1] = self.state[2, 0, -1::-1]
        _state[2, 0] = self.state[4, 1, -1::-1]
        _state[4, 1] = self.state[0, 1]
        if VERBOSE or verbose:
            self.print_transformation(_state)
        self.state = _state

    def B(self, verbose=False):
        if VERBOSE or verbose:
            print "B"
        _state = np.copy(self.state)
        _state[3] = np.rot90(self.state[3])
        _state[0, 0] = self.state[4, 0]
        _state[5, 0] = self.state[0, 0]
        _state[2, 1] = self.state[5, 0, -1::-1]
        _state[4, 0] = self.state[2, 1, -1::-1]
        if VERBOSE or verbose:
            self.print_transformation(_state)
        self.state = _state

    def Bprime(self, verbose=False):
        if VERBOSE or verbose:
            print "Bprime"
        _state = np.copy(self.state)
        _state[3] = np.rot90(self.state[3], k=-1)
        _state[0, 0] = self.state[5, 0]
        _state[5, 0] = self.state[2, 1, -1::-1]
        _state[2, 1] = self.state[4, 0, -1::-1]
        _state[4, 0] = self.state[0, 0]
        if VERBOSE or verbose:
            self.print_transformation(_state)
        self.state = _state

    def scramble(self, moves=10):
        for i in range(moves):
            i = np.random.randint(1, 13)
            self.transformations[i]()
            self.sequence += [i]

    def try_sequence(self, sequence):
        tmp = TwoByTwo()
        tmp.state = np.copy(self.state)
        tmp.apply_sequence(sequence)
        if tmp.degree() == 0:
            return True
        else:
            return False

    def delta(self, sequence):
        tmp = TwoByTwo()
        tmp.state = self.state.copy()
        tmp.apply_sequence(sequence)
        d = 0
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                for k in range(self.state.shape[2]):
                    if self.state[i, j, k] != tmp.state[i, j, k]:
                        d += 1
        return d

    def solve(self):
        for i in self.sequence[-1::-1]:
            if i % 2 == 0:
                self.transformations[i - 1]()
            else:
                self.transformations[i + 1]()

    def degree(self):
        temp = TwoByTwo()
        return 24 - sum([1 for obj in  self.state.flat == temp.state.flat if obj])

    def apply_sequence(self, sequence, verbose=False):
        for i in sequence:
            self.transformations[i](verbose=verbose)

class Permuter:
    def __init__(self, max_length=10):
        self.length = 1
        self.permutation = [1]
        self.max_length = max_length

    def __iter__(self):
        return self

    def next(self):
        i = self.length - 1
        flag = False
        while self.permutation[i] == 12:
            flag = True
            self.permutation[i] = 1
            i -= 1
            if i == -1:
                if self.length == self.max_length:
                    raise StopIteration
                self.length += 1
                self.permutation = [1 for j in range(self.length)]
                return self.permutation
        self.permutation[i] += 1
        if flag is True:
            return self.permutation
        return self.permutation


def main_proc(seq, c):
    if 0 < c.delta(seq) < 9:
        return seq

def outputter(seq1):
    if seq1 is None:
        return
    c1 = TwoByTwo()
    if 7 <= c1.delta(seq1) <= 9:
        infile
    elif 4 <= c1.delta(seq1) <= 6:
        delta 2
    c1.apply_sequence(seq1)
    for line in outfile:
        seq2 = [int(s.strip()) for s in line.split(",")]
        c2 = TwoByTwo()
        c2.apply_sequence(seq2)
        if c1 == c2:
            outfile.close()
            return
    s = str(seq1[0])
    for i in range(1, len(seq1)):
        s += ", %d" % seq1[i]
    print seq1
    outfile.seek(0, 2)
    outfile.write("%s\n" % s)
    outfile.close()

def main():
    for p in generator():
        print p

def generator():
    permuter = Permuter(max_length=20)
    for perm in permuter:
        yield perm

if __name__ == "__main__":
    main()
