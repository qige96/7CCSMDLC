def bin_int(integer):
    '''convert integer to binary string'''
    if integer == 0:
        return '0'
    result = []
    while integer != 0:
        quotient, remainer = divmod(integer, 2)
        result.append(str(remainer))
        integer = quotient
    result.reverse()
    return ''.join(result)

def bin_frac(fractional, precision=4):
    '''
    convert a fractional number to binary string
    Example: bin_frac(0.3215, 4)
    '''
    result = ''
    i = 0
    while fractional != 0 and i < precision:
        result += str(int(fractional * 2))
        fractional = fractional * 2 - int(fractional * 2)
        i += 1
    return result

# print(bin_int(25)+'.'+bin_frac(0.3125))

def encode(lo, hi, num, bits):
    '''encode an integer to a binary string within a given range'''
    assert num>=lo and num<=hi, "num out of domain"
    mapping = round(((num-lo) / (hi-lo))* (2**bits - 1))
    return bin_int(mapping).zfill(bits)

def decode(lo, hi, chrom, bits):
    '''decode a binary string to a number within a given range'''
    return int(lo + int(chrom, 2)*((hi-lo)/(2**bits)))

def binary2grey(bstr):
    '''convert an odinary binary string into a gray code form'''
    gstr = bstr[0]
    for i in range(1, len(bstr)):
        gstr += str(int(bstr[i]) ^ int(bstr[i-1]))
    return gstr

def grey2binary(gstr):
    '''convert an gray code string into an odinary binary form'''
    bstr = gstr[0]
    for i in range(1, len(gstr), 1):
        bstr += str(int(bstr[-1]) ^ int(gstr[i]))
    return bstr

# print(binary2grey('1011'))
# print(grey2binary('1110'))

def pmx(par1, par2, cxp1, cxp2):
    '''partially matched crossover'''
    def _pmx(par1, par2, cxp1, cxp2):
        offsp = par1[:cxp1] + par2[cxp1:cxp2] + par1[cxp2:]
        for i in range(len(offsp)):
            if i in range(cxp1, cxp2):
                continue
            if offsp[i] in offsp[cxp1:cxp2]:
                for x2 in par2:    
                    if x2 not in offsp:
                        offsp[i] = x2
                        break
        return offsp
    return _pmx(par1, par2, cxp1, cxp2), _pmx(par2, par1, cxp1, cxp2)

# print(pmx([3,4,6,2,1,5], [4,1,5,3,2,6], 1, 3))

def ox(par1, par2, cxp1, cxp2):
    '''ordered crossover'''
    def _ox(par1, par2, cxp1, cxp2):
        offsp = par1[:cxp1] + par2[cxp1:cxp2] + par1[cxp2:]
        # copy from the second crossover point
        modified_par1 = par1[cxp2:] + par1[:cxp2]   
        for x in par2[cxp1:cxp2]:  
            # Cross out duplicates
            modified_par1.remove(x)
        # copy from the second crossover point
        positions =  list(range(cxp2, len(par1))) + list(range(0, cxp1))  
        for i in range(len(positions)):
            offsp[positions[i]] = modified_par1[i]
        return offsp
    return _ox(par1, par2, cxp1, cxp2), _ox(par2, par1, cxp1, cxp2)

# print(ox([3,4,6,2,1,5], [4,1,5,3,2,6], 2, 4))

def cx(par1, par2):
    '''cycled crossover'''
    def swap(chrom1, chrom2, pos):
        chrom1[pos], chrom2[pos] = chrom2[pos], chrom1[pos]
    def is_duplicated(chrom, pos):
        if chrom.count(chrom[pos]) > 1:
            return True
        return False
    def is_terminate(chrom):
        for i in range(len(chrom)):
            if is_duplicated(chrom, i):
                return False
        return True
    offsp1 = par1.copy()
    offsp2 = par2.copy()
    swap(offsp1, offsp2, 0)
    while not is_terminate(offsp1):
        for i in range(1, len(offsp1)):
            if is_duplicated(offsp1, i):
                swap(offsp1, offsp2, i)
    return offsp1, offsp2

# print(cx([3,4,6,2,1,5], [4,1,5,3,2,6]))

def ref_encode(par):
    '''coding with reference list'''
    coded_par = []
    reflist = list(range(1, len(par)+1))
    for x in par:
        coded_par.append(reflist.index(x)) # code start from 0
        # coded_par.append(reflist.index(x)+1) # code start from 1
        reflist.remove(x)
    return coded_par 

# print(ref_encode([3,4,6,2,1,5]))

def ref_decode(coded_par):
    '''decode a coded parent'''
    par = []
    reflist = list(range(1, len(coded_par)+1))
    for i in coded_par:
        par.append(reflist[i])
        reflist.remove(reflist[i])
    return par

# print(ref_decode(ref_encode([3,4,6,2,1,5])))

def inversion(chrom, start, end):
    substr = chrom[start:end]
    substr.reverse()
    return chrom[:start] + substr + chrom[end:]

# print(inversion([6,1,5,3,2,4], 1, 5))

def insertion(chrom, start, end, pos):
    substr = chrom[start:end]
    displacement = chrom[:start] + chrom[end:]
    return displacement[:pos] + substr + displacement[pos:]

# print(insertion([6,1,5,3,2,4], 2, 5, 1))

def reciprocal(chrom, pos1, pos2):
    replicate = chrom.copy()
    replicate[pos1], replicate[pos2] = replicate[pos2], replicate[pos1]
    return replicate

# print(reciprocal([6,1,5,3,2,4], 1, 4))
