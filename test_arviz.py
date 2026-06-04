import arviz as az
try:
    import arviz_stats as azs
    print("arviz_stats available!")
    print(dir(azs))
except Exception as e:
    print(e)
