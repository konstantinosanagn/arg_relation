import json
from typing import List
from sklearn.model_selection import train_test_split
from argument_relation.data.data_loader import load_comments
from argument_relation.data.data_types import Comment

def save_comments_to_jsonlist(comments: List[Comment], filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        for comment in comments:
            # Convert Comment back to a dictionary
            comment_dict = {
                "id": comment.id,
                "propositions": [
                    {
                        "id": p.id,
                        "type": p.type,
                        "text": p.text,
                        "reasons": p.reasons if p.reasons else None,
                        "evidence": p.evidence if p.evidence else None
                    }
                    for p in comment.propositions
                ]
            }
            f.write(json.dumps(comment_dict) + "\n")

def main():
    filepath = "/home/parklab/data/cdcp_type_edge_annot/cdcp_type_edge_annot.jsonlist"
    comments = load_comments(filepath)

    # Perform the splits:
    # 1) Extract the test set: test is 30%, so train_val is 70%
    train_val, test = train_test_split(comments, test_size=0.3, random_state=42)

    # 2) Extract validation from train_val: we want validation to be 30% of total
    # validation_ratio = 0.3/0.7 ~ 0.42857
    validation_ratio = 0.3 / 0.7
    train, validation = train_test_split(train_val, test_size=validation_ratio, random_state=42)

    # Check sizes (optional)
    print(f"Training set size: {len(train)}")
    print(f"Validation set size: {len(validation)}")
    print(f"Test set size: {len(test)}")

    # Save the splits into separate jsonlist files within the data directory
    save_comments_to_jsonlist(train, "train.jsonlist")
    save_comments_to_jsonlist(validation, "validation.jsonlist")
    save_comments_to_jsonlist(test, "test.jsonlist")

if __name__ == "__main__":
    main()

