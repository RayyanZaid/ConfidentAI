import pandas as pd

csv_file = 'eye_contact_results.csv'
df = pd.read_csv(csv_file)
passed_nonzero_eyecontact = df[(df['PassedInterview'] == True) & (df['EyeContactPct'] != 0)]
if not passed_nonzero_eyecontact.empty:
    avg_eyecontact_pct = passed_nonzero_eyecontact['EyeContactPct'].mean()
    print(f"Average Eye Contact Percent for Passed Interviews (excluding 0% cases): {avg_eyecontact_pct:.2f}%")
else:
    print("No valid data to calculate average eye contact percent.")
