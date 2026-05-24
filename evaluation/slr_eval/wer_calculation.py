import os
# import pdb
from .python_wer_evaluation import wer_calculation


def evaluate(prefix="./", mode="dev", evaluate_dir=None, evaluate_prefix=None,
             output_file=None, output_dir=None, python_evaluate=False,
             triplet=False, csl_daily=False):
    '''
    TODO  change file save path
    '''
    sclite_path = "./software/sclite"
    print(os.getcwd())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(prefix, output_file)
    tmp_path = os.path.join(prefix, "tmp.ctm")
    tmp2_path = os.path.join(prefix, "tmp2.ctm")
    tmp_stm_path = os.path.join(prefix, "tmp.stm")
    out_path = os.path.join(prefix, f"out.{output_file}")

    # Jika evaluate_prefix mengandung "mslr", gunakan preprocess_bisindo.sh, jika tidak gunakan preprocess.sh. Kedua script ini akan memproses output_file menjadi format yang sesuai untuk evaluasi WER. Setelah itu, kita akan menggunakan sclite untuk menghitung WER dengan membandingkan hasil prediksi (tmp2.ctm) dengan ground-truth (tmp.stm). Jika python_evaluate True, kita juga akan menghitung WER menggunakan fungsi wer_calculation() yang sudah diimplementasikan. Jika triplet True, kita juga akan menghitung WER untuk hasil konversi yang disimpan di out_path.replace(".ctm", "-conv.ctm").
    if evaluate_prefix and "mslr" in evaluate_prefix.lower():
        preprocess_script = "preprocess_bisindo.sh"
    else:
        preprocess_script = "preprocess.sh"
    
    # Jalankan script preprocess untuk memproses output_file menjadi format yang sesuai untuk evaluasi WER. Script ini akan menghasilkan file tmp.ctm dan tmp2.ctm yang akan digunakan untuk evaluasi. Setelah itu, kita akan menggunakan sclite untuk menghitung WER dengan membandingkan hasil prediksi (tmp2.ctm) dengan ground-truth (tmp.stm). Jika python_evaluate True, kita juga akan menghitung WER menggunakan fungsi wer_calculation() yang sudah diimplementasikan. Jika triplet True, kita juga akan menghitung WER untuk hasil konversi yang disimpan di out_path.replace(".ctm", "-conv.ctm").
    os.system(f"bash {script_dir}/{preprocess_script} {output_path} {tmp_path} {tmp2_path}")
    # if not csl_daily:
    #     os.system(f"bash {evaluate_dir}/preprocess.sh {prefix + output_file} {prefix}tmp.ctm {prefix}tmp2.ctm")
    # else:
    #     os.system(f"cp {prefix + output_file} {prefix}tmp2.ctm")
    # pdb.set_trace()
    os.system(f"cat {evaluate_dir}/{evaluate_prefix}-{mode}.stm | sort  -k1,1 > {tmp_stm_path}")
    # pdb.set_trace()
    # tmp2.ctm: prediction result; tmp.stm: ground-truth result
    os.system(f"python {script_dir}/mergectmstm.py {tmp2_path} {tmp_stm_path}")
    os.system(f"cp {tmp2_path} {out_path}")
    if python_evaluate:
        ret = wer_calculation(f"{evaluate_dir}/{evaluate_prefix}-{mode}.stm", out_path)
        if triplet:
            wer_calculation(
                f"{evaluate_dir}/{evaluate_prefix}-{mode}.stm",
                out_path,
                out_path.replace(".ctm", "-conv.ctm")
            )
        return ret
    if output_dir is not None:
        output_dir_path = os.path.join(prefix, output_dir)
        if not os.path.isdir(output_dir_path):
            os.makedirs(output_dir_path)
        os.system(
            f"{sclite_path}  -h {out_path} ctm"
            f" -r {tmp_stm_path} stm -f 0 -o sgml sum rsum pra -O {output_dir_path}"
        )
    else:
        os.system(
            f"{sclite_path}  -h {out_path} ctm"
            f" -r {tmp_stm_path} stm -f 0 -o sgml sum rsum pra"
        )
    ret = os.popen(
        f"{sclite_path}  -h {out_path} ctm "
        f"-r {tmp_stm_path} stm -f 0 -o dtl stdout |grep Error"
    ).readlines()[0]
    # pdb.set_trace()
    return float(ret.split("=")[1].split("%")[0])


if __name__ == "__main__":
    evaluate("output-hypothesis-dev.ctm", mode="dev")
    evaluate("output-hypothesis-test.ctm", mode="test")
