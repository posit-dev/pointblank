import pointblank as pb

validation = (
    pb.Validate(
        data="worldcities.csv",
        thresholds=pb.Thresholds(
            warning=1,  # 1 failure
            error=0.05,  # 5% of rows failing
        ),
    )
    .col_schema_match(
        schema=pb.Schema(
            columns=[
                ("city_name", "object"),
                ("latitude", "float64"),
                ("longitude", "float64"),
                ("country", "object"),
                ("population", "float64"),
            ]
        ),
    )
    .col_vals_not_null(columns="city_name")
    .col_vals_not_null(columns="population")
    .col_vals_gt(columns="population", value=0, na_pass=True)
    .col_vals_between(columns="latitude", left=-90, right=90)
    .col_vals_between(columns="longitude", left=-180, right=180)
    .interrogate()
)
