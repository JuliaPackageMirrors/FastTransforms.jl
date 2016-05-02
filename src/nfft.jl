function nfft( y::Array{Float64,1}, v::Array{Float64,1} )
# A Taylor-based nonuniform fast Fourier transform.
#
#  y = nonuniform points that satisfy 
#            | y[k+1] - k/n | < 1/n/2/pi. 
#  v = input vector 
#  x = output vector
#
# A. Townsend, May 2016. 
# "The world's simplest NFFT" 

    n = size( v, 1 )                   # size of input
    freq = collect(0:n-1)./n           # uniform samples
    D1 = spdiagm( y - freq, 0, n, n )  # nonuniformity
    D2 = spdiagm( convert(Array{Float64,1}, collect(0:n-1)), 0, n, n ) 
    x = fft(v)                     
    for j = 1:18                    # Apply Taylor to each entry
        scl = (-1im)^j/prod( (1/2:1/2:j/2)/pi )  # scl = (-2*pi*1im)^j/factorial(j)
        x = x + scl*D1^j * fft( D2^j*v )     
    end
    x
    
end