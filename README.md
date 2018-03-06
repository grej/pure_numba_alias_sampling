# A faster, more performant Alias Sampling Implementation than numpy's random.choice with probabilities

## Inspired by the algorithm described in the post below, the text by Devroye, and the gist by Jeremy Howard

https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

https://gist.github.com/jph00/30cfed589a8008325eae8f36e2c5b087

http://luc.devroye.org/rnbookindex.html

## Usage

```
from numba_alias_sampling import prep_var_sample, draw_one, draw_n

alias_sample_data = prep_var_sample(np.array([0.11, 0.09, 0.05, 0.75]))
draw_n(50, alias_sample_data)
>>> array([0, 3, 0, 3, 2, 3, 3, 2, 0, 3, 2, 1, 3, 0, 3, 3, 3, 2, 3, 3, 1, 3, 1,
       3, 3, 3, 1, 3, 3, 1, 3, 0, 3, 3, 3, 3, 1, 3, 0, 3, 1, 3, 0, 3, 3, 3,
       0, 1, 3, 3], dtype=int32)
draw_one(alias_sample_data)
>>> 3
```
### Timings
```
%timeit sample_data = prep_var_sample(np.array([0.11, 0.09, 0.05, 0.75])); draw_n(50, sample_data)
>>> 7.53 µs ± 170 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
```
```
%timeit np.random.choice(np.array([0,1,2,3]), p=np.array([0.11, 0.09, 0.05, 0.75]), size=50)
>>> 32.4 µs ± 1.07 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```
