using IterativeSolvers

export nfft, nifft, nfft_transpose

# Define type for cg call: 
type NormalEqns
    nonequiPoints
end
import Base.* 
function *(A::NormalEqns, v::Vector)
     nfft_transpose( A.nonequiPoints, nfft(A.nonequiPoints, v)/n)
end
Base.eltype(A::NormalEqns) = Float64 
Base.size(A::NormalEqns, i::Int) = length(A.nonequiPoints)

function nfft( y::Array{Float64,1}, v::Vector )
# A Taylor-based nonuniform fast Fourier transform. 
# Alex Townsend, May 2016. 

    n = size( v, 1 )                   # size of input
    freq = collect(0:n-1)./n           # uniform samples
    D1 = spdiagm( y - freq, 0, n, n )  # nonuniformity
    D2 = spdiagm( convert(Array{Float64,1}, collect(0:n-1)), 0, n, n ) 
    x = fft( v )                     
    for j = 1:18                    # Apply Taylor to each entry
        scl = (-1im)^j/prod( (1/2:1/2:j/2)/pi )  # scl = (-2*pi*1im)^j/factorial(j)
        x = x + scl*D1^j * fft( D2^j*v )     
    end
    x
end

function nifft( y, v )
# A Taylor-based nonuniform inverse fast Fourier transform.
#
# 
# Alex Townsend, October 2015. 

n = length(v)
# Conjugate gradient on the normal equations: 
B = NormalEqns(y)
x = cg(B, nfft_transpose(y, v)/n)
return x[1]  
end

function nfft_transpose( y, v::Vector )
# A Taylor-based nonuniform transposed fast Fourier transform.
#
#  y = nonuniform points that satisfy 
#            | y[k+1] - k/n | < 1/n/2/pi. 
#  v = input vector 
#  x = output vector
# 
# Alex Townsend, May 2016. 

    n = size( v, 1 )                  # size of input
    freq = collect(0:n-1)./n                # uniform samples
    D1 = spdiagm( y - freq, 0, n, n ) # nonuniformity
    D2 = spdiagm( convert(Array{Float64,1},collect(0:n-1)), 0, n, n )  
    x = n*ifft( v )                    # DFT' = n*IDFT    
    for j = 1:18                       # Apply Taylor to each entry
        scl = (-1im)^j/prod( (1/2:1/2:j/2)/pi )  # scl = (-2*pi*1im)^j/factorial(j)        
        x = x + conj(scl)*D2^j * n * ifft( D1^j*v )
    end
    x
end