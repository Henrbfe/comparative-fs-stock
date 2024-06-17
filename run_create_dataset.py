import sys
from data.create_dataset import create_dataset_given_country


if __name__ == "__main__":
    country_code = sys.argv[1] if len(sys.argv) > 1 else "JPN"
    start_date = sys.argv[2] if len(sys.argv) > 2 else "2022-01-01"
    end_date = sys.argv[3] if len(sys.argv) > 3 else "2024-01-01"
    filename = sys.argv[4] if len(sys.argv) > 4 else f"{country_code}_{start_date}_{end_date}"

    df = create_dataset_given_country(country_code, (start_date, end_date))
    df.to_csv(f"datasets/{filename}.csv")
    print(f"Saved to datasets/{filename}.csv")
