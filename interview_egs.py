# Find repeated value in an array 0..n,  with 1 repeat, in order

d1 = [0,1,2,3,4,5,6,7,8,9,10,11,11,12,13,14]
d2 = [0,0,1,2,3]
d3 = [0,1,2,3,4,4]
    
#n  0 1 2 3 4  
#D  0 0 1 2 3 
#mid = 3
#l = 3 r = 7
#mid = 5,
#l = mid = 5
#mid = 6
#now not equal, so move r
#r=6

#n  0 1 2 3 4 5 6 7
#X  0 1 1 2 3 4 5 6
#l = 0, r = 7
#mid = 3
#move r, r = mid = 3
#l=0, r = 3
#mid = 1
#equal, so l = 1
#mid = 2
#not equal, so r = 2


def fb(d):
    l,r = 0,len(d)-1
    while l < r-1:
        mid = (l+r)//2
        if d[mid] == d[mid+1]:
            return d[mid]
        
        if d[mid] == mid:
            l = mid
        else:
            r = mid
        
    if d[mid] == d[mid+1]:
        return d[mid]
    elif d[mid] == d[mid-1]:
        return d[mid]
    else:
        print (l,r)
        return -1

print (fb(d1))
print (fb(d2))
print (fb(d3))
