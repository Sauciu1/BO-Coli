import pandas as pd



def hit_stats(hit_df):
    output_dict = {}
    for id, df in hit_df.items():
        wd = {
            "TP_index": df[df["assumed_hit"] & df["true_hit"]].index.tolist(),
            "FP_index": df[df["assumed_hit"] & ~df["true_hit"]].index.tolist(),
            "TN_index": df[~df["assumed_hit"] & ~df["true_hit"]].index.tolist(),
            "FN_index": df[~df["assumed_hit"] & df["true_hit"]].index.tolist(),
        }
    
        wd =  wd |{
            "TP": len(wd["TP_index"]),
            "FP": len(wd["FP_index"]),
            "TN": len(wd["TN_index"]),
            "FN": len(wd["FN_index"]),
        }

        total = wd["TP"] + wd["FP"] + wd["TN"] + wd["FN"]

        def sdiv(num, den):
            return num / den if den != 0 else 0.0

        wd = wd | {
            "Precision": sdiv(wd["TP"], wd["TP"] + wd["FP"]),
            "Recall": sdiv(wd["TP"], wd["TP"] + wd["FN"]),
            "Specificity": sdiv(wd["TN"], wd["TN"] + wd["FP"]),
            "Accuracy": sdiv(wd["TP"] + wd["TN"], total),
   
        }
        wd = wd |{
            "F1": sdiv(2 * wd["Precision"] * wd["Recall"], wd["Precision"] + wd["Recall"]),
 
        }

        output_dict[id] = wd


        df= pd.DataFrame(output_dict).T
        df.reset_index(inplace=True, drop=False)

    return df