import os

def split_path(path):
    out = []
    head = True
    while path:
        path, tail = os.path.split(path)
        out.append(tail)
    out.reverse()
    return out

def mkdirpath(path):
    total = path[0]
    if not os.path.isdir(total):
        os.mkdir(total)
    for p in path[1:]:
        total = os.path.join(total, p)
        if not os.path.isdir(total):
            os.mkdir(total)

def all_files(folder):
    return [os.path.join(path, filename) for path, dirs, files in os.walk(folder) for filename in files]


class Flip_Flopper:
    def __init__(self, *flip_flop, names=0):
        self.states = []
        self.names = []
        try:
            for state in flip_flop:
                name, names, state = self.build(state, names)
                self.states.append(state)
                self.names.append(name)
        except TypeError:
            raise TypeError('Nodes must be iterable or int')

        self.state = 0
        self.timer = 0
        self.next_name = names

    def build(self, item, names):
        if type(item) == int:
            if item <= 0:
                raise ValueError('Leaf must have value >= 0')
            if type(names) == int:
                next_name = names + 1
                name = names
            else:
                next_name = names[1:]
                name = names[0]
            return name, next_name, item
        else:
            node = Flip_Flopper(*item, names=names)
            return None, node.next_name, node

    def flip(self):
        return self._flip()[1]

    def _flip(self):
        curr = self.states[self.state]
        if type(curr) == Flip_Flopper:
            flip, name = curr._flip()
            if flip:
                self.state = (self.state + 1)%len(self.states)
            return flip, name
        else:
            self.timer += 1
            name = self.names[self.state]
            if self.timer >= curr:
                self.timer = 0
                self.state = (self.state + 1)%len(self.states)
                return True, name
            else:
                return False, name

    def __str__(self):
        strings = [str(state) for state in self.states]
        for i in range(len(strings)):
            if type(self.states[i]) == int:
                strings[i] = 'X' if i == self.state else 'O'
            else:
                strings[i] = ('X' if i == self.state else 'O') + strings[i]
        return '({})'.format(', '.join(strings))
