# packages
using StatsBase
using MLJ
using MLJModels
using Random
using CategoricalArrays
using PrettyPrinting
using DataFrames
using LossFunctions
using XGBoost
using MLJXGBoostInterface
using CSV
using Gadfly
using Missings

# import data set
data = CSV.File("data/imputed_data.csv", normalizenames = true) |> DataFrame
## Separate the dataset by Dpi 
data_9 = filter(row -> row.Dpi == 9, data)
data_21 = filter(row -> row.Dpi == 21, data)


# Feature selection (Worm Burden, Dpi = 9)
X = data_9[!, [9; 13:27; 42:43; 45:49; 52:end-4]]
y = data_9[!, end-3]
show(stdout, MIME("text/plain"), MLJ.schema(X))
show(y)


perm = randperm(length(y))
X = X[perm,:]
y = y[perm];
train, test = partition(eachindex(y), 0.70, shuffle=true, rng=25)
## check train set balance
plot((data_9[train, :]),
     xgroup = :Line, 
     x = :Group, 
     color = :Protein, 
     Geom.subplot_grid(Geom.histogram),
     Guide.xlabel(""),
     Guide.ylabel("Count"))


models(matching(X, y))

## XGBoost
@load XGBoostCount
xgb_model = XGBoostCount()

xgbm = machine(xgb_model, X, y)
report_full = CSV.File("./results/selected_features_immunology_Dpi_9_worm.csv") |> DataFrame
report_full = report_full[:, 1:2]
for i in 1:1:99
     ## Tuning max_depth and min_child_weight
     r1_1 = range(xgb_model, :max_depth, lower = 3, upper = 10)
     r1_2 = range(xgb_model, :min_child_weight, lower = 1, upper = 5)
     tm = TunedModel(model = xgb_model, tuning = RandomSearch(),
                     resampling=CV(nfolds = 10), ranges=[r1_1, r1_2],
                     measure = rms)
     mtm = machine(tm, X, y)
     fit!(mtm, rows = train)
     xgb_model = fitted_params(mtm).best_model
     @show xgb_model.max_depth
     @show xgb_model.min_child_weight
     ## Tuning gamma
     xgbm = machine(xgb_model, X, y)
     r_2 = range(xgb_model, :gamma, lower=0, upper=10)
     tm = TunedModel(model = xgb_model, tuning = RandomSearch(),
                resampling=CV(nfolds = 10), ranges=r_2,
                measure = rms)
     mtm = machine(tm, X, y)
     fit!(mtm, rows = train)

     xgb_model = fitted_params(mtm).best_model
     @show xgb_model.gamma
     ## Tuning subsample and colsample_bytree
     r3_1 = range(xgb_model, :subsample, lower=0.6, upper=1.0)
     r3_2 = range(xgb_model, :colsample_bytree, lower=0.6, upper=1.0)
     tm = TunedModel(model=xgb_model, tuning=RandomSearch(),
                     resampling=CV(nfolds = 10), ranges=[r3_1,r3_2],
                     measure=rms)
     mtm = machine(tm, X, y)
     fit!(mtm, rows=train)
     xgb_model = fitted_params(mtm).best_model
     @show xgb_model.subsample
     @show xgb_model.colsample_bytree
     ## Tuning alpha & lambda
     r4_1 = range(xgb_model, :alpha, lower=0.6, upper=1.0)
     r4_2 = range(xgb_model, :lambda, lower=0.6, upper=1.0)
     tm = TunedModel(model=xgb_model, tuning=RandomSearch(),
                     resampling=CV(nfolds = 10), ranges=[r4_1,r4_2],
                     measure=rms)
     mtm = machine(tm, X, y)
     fit!(mtm, rows=train)
     xgb_model = fitted_params(mtm).best_model
     @show xgb_model.alpha
     @show xgb_model.lambda
     ## Tuning num_round and eta
     r5_1 = range(xgb_model, :num_round, lower=100, upper=1000)
     r5_2 = range(xgb_model, :eta, lower=0.01, upper=0.1)
     tm = TunedModel(model=xgb_model, tuning=RandomSearch(),
                     resampling=CV(nfolds = 10), ranges=[r5_1,r5_2],
                     measure=rms)
     mtm = machine(tm, X, y)
     fit!(mtm, rows=train)
     xgb_model = fitted_params(mtm).best_model
     @show xgb_model.num_round
     @show xgb_model.eta
     ## Performance test
     tuned_xgb = machine(xgb_model, X, y)
     fit!(tuned_xgb, rows=train, verbosity = 0)
     ŷ = MLJ.predict(tuned_xgb, rows=test)
     y_train = MLJ.predict(tuned_xgb, rows=train)
     rms(y_train, y[train])
     rms(ŷ, y[test])
     ## Plot the residulas in test set
     res = y[test] - ŷ
     points = 1:length(y[test]) |> collect 
     Gadfly.plot(x = points, 
                 y = res, 
                 color = res.^2,
                 Geom.hair, 
                 Geom.point, 
                 Guide.xlabel(""), 
                 Guide.ylabel("Residual (ŷ - y)"), 
                 Theme(key_position = :none))
     ## Feature selection
     report_xgb = report(tuned_xgb)
     report_xgb = report_xgb.feature_importances |> DataFrame
     report_xgb = report_xgb[:, 1:2]
     report_full = outerjoin(report_full, report_xgb, on = :fname, makeunique=true)
end

report_full.average_gain = mean.(Missings.replace.(eachrow(report_full[:, 2:12]), 0))
sort!(report_full, :average_gain, rev = true)
CSV.write("./results/selected_features_full_Dpi_9_worm.csv", report_full)

report_xgb_sel = report_full[1:10, :]
p1 = Gadfly.plot(report_xgb_sel,
                 y = :fname,
                 x = :average_gain,
                 color = :average_gain,
                 Geom.bar(orientation=:horizontal),
                 Guide.xlabel("Gain"),
                 Guide.ylabel("Feature"),
                 Guide.title("Top 10 Feature Importance \n (Worm Bueden, Dpi = 9)"),
                 Scale.color_continuous(colormap=Scale.lab_gradient("lightgreen", "red")),
                 Theme(bar_spacing = 1.5mm, key_position=:none))


# Feature selection (Worm Burden, Dpi = 21)
X = data_21[!, [9; 13:27; 42:43; 45:49; 52:end-4]]
y = data_21[!, end-3]
show(stdout, MIME("text/plain"), MLJ.schema(X))
show(y)

perm = randperm(length(y))
X = X[perm,:]
y = y[perm];
train, test = partition(eachindex(y), 0.70, shuffle=true, rng=13)
## check train set balance
plot((data_21[train, :]),
     xgroup = :Line, 
     x = :Group, 
     color = :Protein, 
     Geom.subplot_grid(Geom.histogram),
     Guide.xlabel(""),
     Guide.ylabel("Count"))


models(matching(X, y))

## XGBoost
@load XGBoostCount
xgb_model = XGBoostCount()

xgbm = machine(xgb_model, X, y)
report_full = CSV.File("./results/selected_features_immunology_Dpi_21_worm.csv") |> DataFrame
report_full = report_full[:, 1:2]
for i in 1:1:99
     ## Tuning max_depth and min_child_weight
     r1_1 = range(xgb_model, :max_depth, lower = 3, upper = 10)
     r1_2 = range(xgb_model, :min_child_weight, lower = 1, upper = 5)
     tm = TunedModel(model = xgb_model, tuning = RandomSearch(),
                     resampling=CV(nfolds = 10), ranges=[r1_1, r1_2],
                     measure = rms)
     mtm = machine(tm, X, y)
     fit!(mtm, rows = train)
     xgb_model = fitted_params(mtm).best_model
     @show xgb_model.max_depth
     @show xgb_model.min_child_weight
     ## Tuning gamma
     xgbm = machine(xgb_model, X, y)
     r_2 = range(xgb_model, :gamma, lower=0, upper=10)
     tm = TunedModel(model = xgb_model, tuning = RandomSearch(),
                resampling=CV(nfolds = 10), ranges=r_2,
                measure = rms)
     mtm = machine(tm, X, y)
     fit!(mtm, rows = train)

     xgb_model = fitted_params(mtm).best_model
     @show xgb_model.gamma
     ## Tuning subsample and colsample_bytree
     r3_1 = range(xgb_model, :subsample, lower=0.6, upper=1.0)
     r3_2 = range(xgb_model, :colsample_bytree, lower=0.6, upper=1.0)
     tm = TunedModel(model=xgb_model, tuning=RandomSearch(),
                     resampling=CV(nfolds = 10), ranges=[r3_1,r3_2],
                     measure=rms)
     mtm = machine(tm, X, y)
     fit!(mtm, rows=train)
     xgb_model = fitted_params(mtm).best_model
     @show xgb_model.subsample
     @show xgb_model.colsample_bytree
     ## Tuning alpha & lambda
     r4_1 = range(xgb_model, :alpha, lower=0.6, upper=1.0)
     r4_2 = range(xgb_model, :lambda, lower=0.6, upper=1.0)
     tm = TunedModel(model=xgb_model, tuning=RandomSearch(),
                     resampling=CV(nfolds = 10), ranges=[r4_1,r4_2],
                     measure=rms)
     mtm = machine(tm, X, y)
     fit!(mtm, rows=train)
     xgb_model = fitted_params(mtm).best_model
     @show xgb_model.alpha
     @show xgb_model.lambda
     ## Tuning num_round and eta
     r5_1 = range(xgb_model, :num_round, lower=100, upper=1000)
     r5_2 = range(xgb_model, :eta, lower=0.01, upper=0.1)
     tm = TunedModel(model=xgb_model, tuning=RandomSearch(),
                     resampling=CV(nfolds = 10), ranges=[r5_1,r5_2],
                     measure=rms)
     mtm = machine(tm, X, y)
     fit!(mtm, rows=train)
     xgb_model = fitted_params(mtm).best_model
     @show xgb_model.num_round
     @show xgb_model.eta
     ## Performance test
     tuned_xgb = machine(xgb_model, X, y)
     fit!(tuned_xgb, rows=train, verbosity = 0)
     ŷ = MLJ.predict(tuned_xgb, rows=test)
     y_train = MLJ.predict(tuned_xgb, rows=train)
     rms(y_train, y[train])
     rms(ŷ, y[test])
     ## Plot the residulas in test set
     res = y[test] - ŷ
     points = 1:length(y[test]) |> collect 
     Gadfly.plot(x = points, 
                 y = res, 
                 color = res.^2,
                 Geom.hair, 
                 Geom.point, 
                 Guide.xlabel(""), 
                 Guide.ylabel("Residual (ŷ - y)"), 
                 Theme(key_position = :none))
     ## Feature selection
     report_xgb = report(tuned_xgb)
     report_xgb = report_xgb.feature_importances |> DataFrame
     report_xgb = report_xgb[:, 1:2]
     report_full = outerjoin(report_full, report_xgb, on = :fname, makeunique=true)
end


report_full.average_gain = mean.(Missings.replace.(eachrow(report_full[:, 2:12]), 0))
sort!(report_full, :average_gain, rev = true)
CSV.write("./results/selected_features_full_Dpi_21_worm.csv", report_full)

report_xgb_sel = report_full[1:10, :]
p2 = Gadfly.plot(report_xgb_sel,
                 y = :fname,
                 x = :average_gain,
                 color = :average_gain,
                 Geom.bar(orientation=:horizontal),
                 Guide.xlabel("Gain"),
                 Guide.ylabel("Feature"),
                 Guide.title("Top 10 Feature Importance \n (Worm Bueden, Dpi = 21)"),
                 Scale.color_continuous(colormap=Scale.lab_gradient("lightgreen", "red")),
                 Theme(bar_spacing = 1.5mm, key_position=:none))





# Feature selection (EIC, Dpi = 21)
X = data_21[!, [9; 13:27; 42:43; 45:49; 52:end-4]]
y = data_21[!, end-1]
show(stdout, MIME("text/plain"), MLJ.schema(X))
show(y)


perm = randperm(length(y))
X = X[perm,:]
y = y[perm];
train, test = partition(eachindex(y), 0.70, shuffle=true, rng=13)
## check train set balance
plot((data_21[train, :]),
     xgroup = :Line, 
     x = :Group, 
     color = :Protein, 
     Geom.subplot_grid(Geom.histogram),
     Guide.xlabel(""),
     Guide.ylabel("Count"))



models(matching(X, y))

## XGBoost
@load XGBoostRegressor
xgb_model = XGBoostRegressor()

xgbm = machine(xgb_model, X, y)
report_full = CSV.File("./results/selected_features_immunology_Dpi_21_EIC.csv") |> DataFrame
report_full = report_full[:, 1:2]

for i in 1:1:99
## Tuning max_depth and min_child_weight
r1_1 = range(xgb_model, :max_depth, lower = 3, upper = 10)
r1_2 = range(xgb_model, :min_child_weight, lower = 1, upper = 5)
tm = TunedModel(model = xgb_model, tuning = RandomSearch(),
                resampling=CV(nfolds = 10), ranges=[r1_1, r1_2],
                measure = rms)
mtm = machine(tm, X, y)

fit!(mtm, rows = train)

xgb_model = fitted_params(mtm).best_model
@show xgb_model.max_depth
@show xgb_model.min_child_weight

## Tuning gamma
xgbm = machine(xgb_model, X, y)
r_2 = range(xgb_model, :gamma, lower=0, upper=10)
tm = TunedModel(model = xgb_model, tuning = RandomSearch(),
                resampling=CV(nfolds = 10), ranges=r_2,
                measure = rms)
mtm = machine(tm, X, y)
fit!(mtm, rows = train)

xgb_model = fitted_params(mtm).best_model
@show xgb_model.gamma


## Tuning subsample and colsample_bytree
r3_1 = range(xgb_model, :subsample, lower=0.6, upper=1.0)
r3_2 = range(xgb_model, :colsample_bytree, lower=0.6, upper=1.0)
tm = TunedModel(model=xgb_model, tuning=RandomSearch(),
                resampling=CV(nfolds = 10), ranges=[r3_1,r3_2],
                measure=rms)
mtm = machine(tm, X, y)
fit!(mtm, rows=train)

xgb_model = fitted_params(mtm).best_model
@show xgb_model.subsample
@show xgb_model.colsample_bytree

## Tuning alpha & lambda
r4_1 = range(xgb_model, :alpha, lower=0.6, upper=1.0)
r4_2 = range(xgb_model, :lambda, lower=0.6, upper=1.0)
tm = TunedModel(model=xgb_model, tuning=RandomSearch(),
                resampling=CV(nfolds = 10), ranges=[r4_1,r4_2],
                measure=rms)
mtm = machine(tm, X, y)
fit!(mtm, rows=train)

xgb_model = fitted_params(mtm).best_model
@show xgb_model.alpha
@show xgb_model.lambda

## Tuning num_round and eta

r5_1 = range(xgb_model, :num_round, lower=100, upper=1000)
r5_2 = range(xgb_model, :eta, lower=0.01, upper=0.1)
tm = TunedModel(model=xgb_model, tuning=RandomSearch(),
                resampling=CV(nfolds = 10), ranges=[r5_1,r5_2],
                measure=rms)
mtm = machine(tm, X, y)
fit!(mtm, rows=train)

xgb_model = fitted_params(mtm).best_model
@show xgb_model.num_round
@show xgb_model.eta

## Performance test
tuned_xgb = machine(xgb_model, X, y)
fit!(tuned_xgb, rows=train, verbosity = 0)

ŷ = MLJ.predict(tuned_xgb, rows=test)
y_train = MLJ.predict(tuned_xgb, rows=train)
rms(y_train, y[train])
rms(ŷ, y[test])

## Plot the residulas in test set
res = y[test] - ŷ
points = 1:length(y[test]) |> collect 
Gadfly.plot(x = points, 
            y = res, 
            color = res.^2,
            Geom.hair, 
            Geom.point, 
            Guide.xlabel(""), 
            Guide.ylabel("Residual (ŷ - y)"), 
            Theme(key_position = :none))

## Feature selection
report_xgb = report(tuned_xgb)
report_xgb = report_xgb.feature_importances |> DataFrame
report_xgb = report_xgb[:, 1:2]
report_full = outerjoin(report_full, report_xgb, on = :fname, makeunique=true)
end

report_full.average_gain = mean.(Missings.replace.(eachrow(report_full[:, 2:12]), 0))
sort!(report_full, :average_gain, rev = true)
CSV.write("./results/selected_features_full_Dpi_21_EIC.csv", report_full)

report_xgb_sel = report_full[1:10, :]
p2 = Gadfly.plot(report_xgb_sel,
                 y = :fname,
                 x = :average_gain,
                 color = :average_gain,
                 Geom.bar(orientation=:horizontal),
                 Guide.xlabel("Gain"),
                 Guide.ylabel("Feature"),
                 Guide.title("Top 10 Feature Importance \n (EIC, Dpi = 21)"),
                 Scale.color_continuous(colormap=Scale.lab_gradient("lightgreen", "red")),
                 Theme(bar_spacing = 1.5mm, key_position=:none))