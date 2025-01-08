import pandas as pd
import argparse
from augment import load_tuned_model,augment_one_sent

def main():
    # 모델 및 토크나이저 로딩
    args = argparse.Namespace(
        tuned_model_path="seoyeon96/KcELECTRA-MLM",
        dev_num=0,
        input_file=None,
        batch_size=1,
        mlm_prob=0.15,
        threshold=0.95,
        k=5
    )
    
    model, tokenizer, dev = load_tuned_model(args)
    
    # 데이터 로딩
    path = ""
    df = pd.read_excel(path)
    
    # "prompt" 및 "competition" 열의 텍스트 증강
    augmented_data = []
    for _, row in df.iterrows():
        for column in ["prompt", "competition"]:
            # Convert the data to string type before passing
            text_data = str(row[column])
            augmented_text = augment_one_sent(model, tokenizer, text_data, dev, args)
            row[column] = augmented_text
        augmented_data.append(row)
    
    
    # 증강된 데이터와 원본 데이터 합치기
    augmented_df = pd.DataFrame(augmented_data)
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    combined_df = combined_df.drop(['Unnamed: 2'], axis = 1)
    
    print(combined_df)
    
    # 엑셀 파일로 저장
    aug_file_name = ""
    combined_df.to_csv(aug_file_name, index=False)

if __name__ == '__main__':
    main()