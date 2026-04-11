from Bio import SeqIO

def compare_fasta_headers(fasta_file1, fasta_file2):
    # 读取第一个FASTA文件的表头列表
    headers1 = []
    with open(fasta_file1, 'r') as handle1:
        for record in SeqIO.parse(handle1, 'fasta'):
            headers1.append(record.description.strip())

    # 读取第二个FASTA文件的表头列表
    headers2 = []
    with open(fasta_file2, 'r') as handle2:
        for record in SeqIO.parse(handle2, 'fasta'):
            headers2.append(record.description.strip())

    # 比较两个表头列表是否相同
    if headers1 == headers2:
        print("两个FASTA文件中的蛋白质名称完全相同。")
    else:
        print("两个FASTA文件中的蛋白质名称不相同。")
        # 找出不同的部分
        set1 = set(headers1)
        set2 = set(headers2)
        only_in_file1 = set1 - set2
        only_in_file2 = set2 - set1

        if only_in_file1:
            print("只在第一个文件中存在的蛋白质名称：")
            for header in only_in_file1:
                print(header)
        if only_in_file2:
            print("只在第二个文件中存在的蛋白质名称：")
            for header in only_in_file2:
                print(header)


compare_fasta_headers("/data2/guest1/newtest/mix.fasta","/data2/guest1/newtest/mix_mutated.fasta")