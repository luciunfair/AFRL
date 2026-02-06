#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.special import gammaincc
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm

def approximate_entropy_test(byte_data):
    """
    Approximate Entropy Test (NIST SP 800-22)
     Optimized with NumPy.
    
    Args:
        byte_data: A list or numpy array of integers (0-255).
                   Example: [255, 0, 128, ...]
    
    Returns:
        (success, p, None)
    """
    # Convert bytes (0-255) to a bit array (0-1) efficiently
    # np.unpackbits unpacks uint8 elements into bits (big-endian order)
    data = np.array(byte_data, dtype=np.uint8)
    bits = np.unpackbits(data)
    n = len(bits)
    
    # Calculate m based on the length of the bit stream
    m = int(np.floor(np.log2(n))) - 6
    if m < 2:
        m = 2
    if m > 3:
        m = 3
    
    phi_m = []
    
    # Loop for block sizes m and m+1
    for iterm in range(m, m + 2):
        # Step 1: Pad the bits (Circular buffer logic)
        # We append the first iterm-1 bits to the end
        padded_bits = np.concatenate((bits, bits[:iterm-1]))
        
        # Step 2: Vectorized Pattern Counting
        # Instead of iterating and converting substrings to ints in a loop,
        # we construct the integer values for all positions at once using bitwise shifts.
        
        # We use a container for the integer values of the patterns
        # Since m is capped at 3, patterns are max 4 bits (values 0-15), so int8 is sufficient.
        vals = np.zeros(n, dtype=np.int8)
        
        # Construct the integers for every window of length 'iterm'
        for k in range(iterm):
            vals <<= 1
            # Add the bit at the specific offset. 
            # Slice padded_bits from k to n+k to get the bit at that position for every window
            vals |= padded_bits[k : n + k]
            
        # Count occurrences of each pattern (0 to 2^iterm - 1)
        # bincount is extremely fast compared to a python loop
        counts = np.bincount(vals, minlength=2**iterm)
        
        # Step 3: Calculate Probabilities (Ci)
        Ci = counts.astype(np.float64) / n
        
        # Step 4: Calculate Phi
        # Formula: sum( Ci * log(Ci / 10.0) )
        # We only compute log for non-zero probabilities to avoid -inf
        valid_Ci = Ci[Ci > 0]
        
        # Note: The original code divides by 10.0 inside the log.
        # While mathematically this constant cancels out in the final subtraction,
        # we keep it to strictly match the original code's operation order.
        phi_val = np.sum(valid_Ci * np.log(valid_Ci / 10.0))
        phi_m.append(phi_val)
        
    # Step 6
    appen_m = phi_m[0] - phi_m[1]
    chisq = 2 * n * (np.log(2) - appen_m)
    
    # Step 7
    # Using scipy.special.gammaincc (Regularized upper incomplete gamma function)
    # This matches the statistical definition used in the NIST test.
    p = gammaincc(2**(m-1), (chisq / 2.0))
    
    success = (p >= 0.01)
    return (success, p, None)

#!/usr/bin/env python
def frequency_within_block_test(byte_data):
    """
    Frequency (Monobit) Test within a Block
    NIST SP 800-22
    Optimized with NumPy
    Input: byte array (values 0–255)
    Output: (success, p, None)
    """

    # Convert bytes → bits (0/1)
    data = np.asarray(byte_data, dtype=np.uint8)
    bits = np.unpackbits(data)
    n = bits.size

    # Original NIST constraint
    if n < 100:
        return False, 1.0, None

    # Block size selection (same logic as original)
    M = 20
    N = n // M
    if N > 99:
        N = 99
        M = n // N

    # Truncate to exact multiple
    bits = bits[:N * M]

    # Reshape into blocks: shape (N, M)
    blocks = bits.reshape((N, M))

    # Count ones per block (vectorized)
    ones = np.sum(blocks, axis=1)

    # Proportion of ones per block
    proportions = ones / float(M)

    # Chi-square calculation (exact same formula)
    chisq = np.sum(4.0 * M * (proportions - 0.5) ** 2)

    # P-value
    p = gammaincc(N / 2.0, chisq / 2.0)

    success = (p >= 0.01)
    return (success, p, None)


#!/usr/bin/env python

def runs_test(byte_data):
    """
    Runs Test
    NIST SP 800-22
    Optimized with NumPy
    Input: byte array (0–255)
    Output: (success, p, None)
    """

    # Convert bytes → bits (0/1)
    data = np.asarray(byte_data, dtype=np.uint8)
    bits = np.unpackbits(data)
    n = bits.size

    # Count ones and zeroes (vectorized)
    ones = np.sum(bits)
    zeroes = n - ones

    # Proportion of ones
    prop = ones / float(n)

    # Tau threshold
    tau = 2.0 / math.sqrt(n)

    # NIST early failure condition
    if abs(prop - 0.5) > tau:
        return (False, 0.0, None)

    # Compute number of runs (V_obs)
    # Count transitions between consecutive bits
    # diff != 0 → transition
    transitions = np.sum(bits[:-1] != bits[1:])
    vobs = 1.0 + transitions

    # P-value calculation (exact same formula)
    numerator = abs(vobs - (2.0 * n * prop * (1.0 - prop)))
    denominator = 2.0 * math.sqrt(2.0 * n) * prop * (1.0 - prop)

    p = math.erfc(numerator / denominator)

    success = (p >= 0.01)
    return (success, p, None)

def local_nibbles_variance(byte_data, window_size=64):
    """
    Feature: Local Nibbles Variance (LNV)
    
    Calculates the variance of the Chi-Square statistic for nibble (4-bit) distributions
    across sliding windows. High variance indicates structure (Compression); 
    Low variance indicates uniformity (Encryption).
    
    Parameters:
        row (numpy.ndarray or list): Input byte fragment (e.g., 4096 bytes).
        window_size (int): Size of the local analysis window (default 64).
        
    Returns:
        float: The variance of local chi-square scores.
    """
    # Ensure input is a numpy array of integers
    data = np.array(byte_data, dtype=np.uint8)
    
    # 1. Validation: Ensure we have enough data for at least one window + variance
    # Variance requires at least 2 data points.
    # Step size is window_size // 2.
    # Minimum length needed = window_size + step_size = 1.5 * window_size
    if len(data) < (window_size * 1.5):
        return 0.0

    # 2. Create sliding windows (Vectorized for speed)
    step = window_size // 2
    # Calculate number of windows
    n_windows = (len(data) - window_size) // step + 1
    
    # Create a view of sliding windows (no data copying, very fast)
    # Shape: (n_windows, window_size)
    # This stride trick is standard numpy for sliding windows
    indexer = np.arange(window_size)[None, :] + np.arange(n_windows)[:, None] * step
    windows = data[indexer]

    # 3. The Nibble Trick (Split bytes into 4-bit chunks)
    # Shape becomes (n_windows, window_size * 2)
    high_nibbles = windows >> 4
    low_nibbles = windows & 0x0F
    nibbles = np.concatenate((high_nibbles, low_nibbles), axis=1)

    # 4. Calculate Chi-Square for each window
    # We have 16 bins (0-15).
    # Expected count (E) per bin = total_nibbles / 16
    n_nibbles = nibbles.shape[1] # Should be 128 for window_size=64
    E = n_nibbles / 16.0         # Should be 8.0
    
    # Count occurrences of 0-15 in each window
    # We use an identity matrix trick to vectorize bincount across rows
    # nibbles is (n_windows, 128) -> values 0..15
    # Result counts is (n_windows, 16)
    rows = np.arange(n_windows)[:, None]
    counts = np.zeros((n_windows, 16), dtype=int)
    np.add.at(counts, (rows, nibbles), 1)

    # Compute Chi-Square for each window: sum((O - E)^2 / E)
    # Result is array of shape (n_windows,)
    chi2_scores = np.sum(((counts - E) ** 2) / E, axis=1)
    float(np.var(chi2_scores))<=46.98
    success = float(np.var(chi2_scores))<=46.981
        
    # 5. Return the Variance of these scores
    return (success, float(np.var(chi2_scores)),  None)

def AFRL_test(byte_data):
    """

    Input: byte array (0–255)
    Output: array of 0 or 1 (o means unecrypted and 1 means encrypted)
    """
    row = np.array(byte_data, dtype=np.uint8)

    r1 = approximate_entropy_test(row)[0]
    r2 = frequency_within_block_test(row)[0]
    r3 = runs_test(row)[0]
    r4 = local_nibbles_variance(row)[0]

    result = (r1 == True) and (r2 == True) and (r3 == True) and (r4 == True)
    return result

if __name__ == "__main__":
    input_addr = input("input csv dataset(each row containing bytes eg[234, 2, ...]) address: ")
    output_addr = input("address of csv file you want the results save to ( 0 means unecrypted, 1 means encrypted): ")
    print("loading dataset ...")
    dataset = pd.read_csv(input_addr, header=None).values
    total = len(dataset)

    start = time.perf_counter()

    results = np.empty(total, dtype=np.uint8)

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(AFRL_test, dataset[i]): i
            for i in range(total)
        }

        for future in tqdm(
            as_completed(futures),
            total=total,
            desc="Processing",
            unit="rows"
        ):
            idx = futures[future]
            results[idx] = future.result()

    end = time.perf_counter()

    print(f"\nExecution time: {end - start:.6f} seconds")
    print("wait it is saving")

    pd.DataFrame(results).to_csv(output_addr, index=False, header=False)