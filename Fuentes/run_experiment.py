import subprocess
import sys
import os
import pandas as pd
import numpy as np

def run_script(script_name):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    try:
        subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except:
        return False

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    run_script('rdim.py')
    
    results = []
    for i in range(5):
        run_script('trn.py')
        run_script('tst.py')
        
        fscores_path = os.path.join(script_dir, 'fscores.csv')
        if os.path.exists(fscores_path):
            df = pd.read_csv(fscores_path, header=None)
            fscore_class0 = df.iloc[0, 0]
            fscore_class1 = df.iloc[1, 0]
            fscore_avg = df.iloc[2, 0]
            results.append([fscore_class0, fscore_class1, fscore_avg])
        else:
            results.append([0.0, 0.0, 0.0])
    
    results_array = np.array(results)
    results_df = pd.DataFrame(results_array, columns=['Class_0', 'Class_1', 'Mean'])
    results_df.to_csv(os.path.join(script_dir, 'results_fscores.csv'), index=False, header=False)

if __name__ == '__main__':
    main()
