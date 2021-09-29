#=
StatsModels:
- Julia version: 1.6
- Authors: Simon Babayan
- Date: 9-05-2021
=#

## Load packages
using StatsBase
using MLJ
using PrettyPrinting
using DataFrames
using CSV
using Gadfly
using GLM
using StatsFuns
using MixedModels
using Formatting
using HypothesisTests
using Cairo

## Functions
"""
Test Interactive effects

Function to identify significant interaction terms in a linear model
"""
function test_interactions(formula_0, data_dpi, model_0)
    for interaction in formula_0.rhs[4:end]
    new_formula = drop_term(formula_0, interaction)
    new_model = fit(LinearModel, new_formula, data_dpi)
    LL = 2*(loglikelihood(model_0) - loglikelihood(new_model))
    df = dof(model_0) - dof(new_model)
    LRT = chisqccdf(df, LL)
    if LRT < 0.05
            print("• Testing \n $formula_0 against \n $new_formula \n[removing $interaction]")
            print("\n 2ΔLL = $LL, df = $df")
            printfmt("\n Likelihood Ratio test: {1:.3f} *", LRT)
        else
            print("• Testing \n $formula_0 against \n $new_formula \n[removing $interaction]")
            print("\n 2ΔLL = $LL, df = $df")
            printfmt("\n Likelihood Ratio test: {1:.3f}", LRT)
        end
    println("\n")
    end
end


"""
Test main effects

Function to identify significant main effect terms in a linear model
"""
function test_direct(formula_0, data_dpi, model_0)
    for fixed in formula_0.rhs[1:end]
    new_formula = drop_term(formula_0, fixed)
    new_model = fit(LinearModel, new_formula, data_dpi)
    LL = 2*(loglikelihood(model_0) - loglikelihood(new_model))
    df = dof(model_0) - dof(new_model)
    LRT = chisqccdf(df, LL)
        if LRT == 1.0
            println("")
        elseif LRT < 0.05
            print("• Testing \n $formula_0 against \n $new_formula \n[removing $fixed]")
            print("\n 2ΔLL = $LL, df = $df")
            printfmt("\n Likelihood Ratio test: {1:.3f} *", LRT)
        else
            print("• Testing \n $formula_0 against \n $new_formula \n[removing $fixed]")
            print("\n 2ΔLL = $LL, df = $df")
            printfmt("\n Likelihood Ratio test: {1:.3f}", LRT)
        end
        println("\n")
    end
end

"""
Test Interactive effects

Function to identify significant interaction terms in a linear model (from the 6th term)
"""
function test_interactions_6(formula_0, data_dpi, model_0)
    for interaction in formula_0.rhs[6:end]
    new_formula = drop_term(formula_0, interaction)
    new_model = fit(LinearModel, new_formula, data_dpi)
    LL = 2*(loglikelihood(model_0) - loglikelihood(new_model))
    df = dof(model_0) - dof(new_model)
    LRT = chisqccdf(df, LL)
    if LRT < 0.05
            print("• Testing \n $formula_0 against \n $new_formula \n[removing $interaction]")
            print("\n 2ΔLL = $LL, df = $df")
            printfmt("\n Likelihood Ratio test: {1:.3f} *", LRT)
        else
            print("• Testing \n $formula_0 against \n $new_formula \n[removing $interaction]")
            print("\n 2ΔLL = $LL, df = $df")
            printfmt("\n Likelihood Ratio test: {1:.3f}", LRT)
        end
    println("\n")
    end
end

## Load dataset
raw_data = CSV.File("data/SB-105-Full.csv", normalizenames = true) |> DataFrame
raw_data = coerce(raw_data, :Dpi => Multiclass,
                            :Mouse => Multiclass,
                            :Line => Multiclass,
                            :Group => Multiclass,
                            :Protein => Multiclass)


## transform IgG1, IgG2a titre, and IgE 
raw_data[:, 11] = replace.(raw_data[:, 11], r"," => "")
raw_data[:, 12] = replace.(raw_data[:, 12], r"," => "")
raw_data.normalized_IgG1_serum = parse.(Float64, raw_data[:, 11])
raw_data.normalized_IgG2a_serum = parse.(Float64, raw_data[:, 12])
raw_data.normalized_IgG1_serum = -log.((raw_data.normalized_IgG1_serum .+ 1).^-1)
raw_data.normalized_IgG2a_serum = -log.((raw_data.normalized_IgG2a_serum .+ 1).^-1)
raw_data.log_EIC = log10.(raw_data.EIC .+ 1)
raw_data.log_worm = log10.(raw_data.Total_worms .+ 1)
describe(raw_data) |> pprint


## Load features of interest
feat_sel = ["_IL_10_CD_MLN_stim"; "_IL_5_CD3_MLN_stim"; "_CD36_QPCR_Spleen"; "_IgE_serum"; "_iNOS_QPCR_Gut"; "_iNOS_QPCR_spleen"]


## Separate the dataset by Dpi 
data_9 = filter(row -> row.Dpi == 9, raw_data)
data_21 = filter(row -> row.Dpi == 21, raw_data)
data_9 |> describe |> pprint
data_21 |> describe |> pprint

# 1. Dpi == 9 
# 1.1 Analyse _CD36_QPCR_Spleen    
### model1 
formula_full = @formula(_CD36_QPCR_Spleen  ~ log_worm + log_EIC);
model0 = fit(LinearModel,
             formula_full,
             data_9);
## LRT tests
test_direct(formula_full, data_9, model0)

model1 = fit(LinearModel,
             @formula(_CD36_QPCR_Spleen  ~ log_EIC + log_worm),
             data_9)
### plot the residuals
residuals(model1) |> JarqueBeraTest
plot(x = residuals(model1), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("_CD36_QPCR_Spleen"))

### model2 
formula = @formula(_CD36_QPCR_Spleen ~ log_worm + log_EIC + Line + Protein + Group + Line&Protein + Line&Group + Protein&Group);
model0 = fit(LinearModel, formula, data_9);
### LRT tests
test_interactions_6(formula, data_9, model0)
formula = @formula(_CD36_QPCR_Spleen ~ log_worm + log_EIC + Line + Protein + Group);
model0 = fit(LinearModel, formula, data_9);
test_direct(formula, data_9, model0)
model2 = fit(LinearModel, 
             @formula(_CD36_QPCR_Spleen ~ log_EIC), 
             data_9)
### plot the residuals
residuals(model2) |> JarqueBeraTest
plot(x = residuals(model2), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("_CD36_QPCR_Spleen"))

### Non-parametric tests
### test Line
KruskalWallisTest(filter(row -> row.Line == "ROH", dropmissing(data_9, :_CD36_QPCR_Spleen))[:, :_CD36_QPCR_Spleen], 
                  filter(row -> row.Line == "ROL", dropmissing(data_9, :_CD36_QPCR_Spleen))[:, :_CD36_QPCR_Spleen])
### test Protein
KruskalWallisTest(filter(row -> row.Protein == "HP", dropmissing(data_9, :_CD36_QPCR_Spleen))[:, :_CD36_QPCR_Spleen], 
                  filter(row -> row.Protein == "LP", dropmissing(data_9, :_CD36_QPCR_Spleen))[:, :_CD36_QPCR_Spleen])
### test Group 
KruskalWallisTest(filter(row -> row.Group == "0_1", dropmissing(data_9, :_CD36_QPCR_Spleen))[:, :_CD36_QPCR_Spleen], 
                  filter(row -> row.Group == "1_1", dropmissing(data_9, :_CD36_QPCR_Spleen))[:, :_CD36_QPCR_Spleen])


# 1.2 Analyse _IL_5_CD3_MLN_stim     
## model1 
formula_full = @formula(_IL_5_CD3_MLN_stim   ~ log_worm + log_EIC);
model0 = fit(LinearModel,
             formula_full,
             data_9);
## LRT tests
test_direct(formula_full, data_9, model0)

model1 = fit(LinearModel,
             @formula(_IL_5_CD3_MLN_stim  ~ log_worm),
             data_9)
### plot the residuals
residuals(model1) |> JarqueBeraTest
plot(x = residuals(model1), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("_IL_5_CD3_MLN_stim"))

### model2 
formula = @formula(_IL_5_CD3_MLN_stim ~ log_worm + log_EIC + Line + Protein + Group + Line&Protein + Line&Group + Protein&Group);
model0 = fit(LinearModel, formula, data_9);
### LRT tests
test_interactions_6(formula, data_9, model0)
formula = @formula(_IL_5_CD3_MLN_stim ~ log_worm + log_EIC + Line + Protein + Group);
model0 = fit(LinearModel, formula, data_9);
test_direct(formula, data_9, model0)
model2 = fit(LinearModel, 
             @formula(_IL_5_CD3_MLN_stim ~ 1), 
             data_9)
### plot the residuals
residuals(model2) |> JarqueBeraTest
plot(x = residuals(model2), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("_IL_5_CD3_MLN_stim"))


# 1.3 Analyse _IL_10_CD_MLN_stim     
## model1 
formula_full = @formula(_IL_10_CD_MLN_stim   ~ log_worm + log_EIC);
model0 = fit(LinearModel,
             formula_full,
             data_9);
## LRT tests
test_direct(formula_full, data_9, model0)

model1 = fit(LinearModel,
             @formula(_IL_10_CD_MLN_stim  ~ 1),
             data_9)
### plot the residuals
residuals(model1) |> JarqueBeraTest
plot(x = residuals(model1), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("_IL_10_CD_MLN_stim"))

### model2 
formula = @formula(_IL_10_CD_MLN_stim ~ log_worm + log_EIC + Line + Protein + Group + Line&Protein + Line&Group + Protein&Group);
model0 = fit(LinearModel, formula, data_9);
### LRT tests
test_interactions_6(formula, data_9, model0)
formula = @formula(_IL_10_CD_MLN_stim ~ log_worm + log_EIC + Line + Protein + Group);
model0 = fit(LinearModel, formula, data_9);
test_direct(formula, data_9, model0)
model2 = fit(LinearModel, 
             @formula(_IL_10_CD_MLN_stim ~ 1), 
             data_9)
### plot the residuals
residuals(model2) |> JarqueBeraTest
plot(x = residuals(model2), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("_IL_10_CD_MLN_stim"))

## 1.4 _IgE_serum 
## model1 
formula_full = @formula(_IgE_serum  ~ log_worm + log_EIC);
model0 = fit(LinearModel,
             formula_full,
             data_9);
## LRT tests
test_direct(formula_full, data_9, model0)

model1 = fit(LinearModel,
             @formula(_IgE_serum  ~ log_worm),
             data_9)
### plot the residuals
residuals(model1) |> JarqueBeraTest
plot(x = residuals(model1), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("_IgE_serum"))

## model2 
formula = @formula(_IgE_serum ~ log_worm + log_EIC + Line + Protein + Group + Line&Protein + Line&Group + Protein&Group);
model0 = fit(LinearModel, formula, data_9);
### LRT tests
test_interactions_6(formula, data_9, model0)
formula = @formula(_IgE_serum ~ log_worm + log_EIC + Line + Protein + Group);
model0 = fit(LinearModel, formula, data_9);
test_direct(formula, data_9, model0)
model2 = fit(LinearModel, 
             @formula(_IgE_serum ~ Line + Group), 
             data_9)
### plot the residuals
residuals(model2) |> JarqueBeraTest
plot(x = residuals(model2), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("IgE"))


## 1.5 ADWG 
## model1 
formula_full = @formula(ADWG  ~ log_worm + log_EIC);
model0 = fit(LinearModel,
             formula_full,
             data_9);
## LRT tests
test_direct(formula_full, data_9, model0)

model1 = fit(LinearModel,
             @formula(ADWG  ~ 1),
             data_9)
### plot the residuals
residuals(model1) |> JarqueBeraTest
plot(x = residuals(model1), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("ADWG"))

## model2 
formula = @formula(ADWG ~ log_worm + log_EIC + Line + Protein + Group + Line&Protein + Line&Group + Protein&Group);
model0 = fit(LinearModel, formula, data_9);
### LRT tests
test_interactions_6(formula, data_9, model0)
formula = @formula(ADWG ~ log_worm + log_EIC + Line + Protein + Group + Line&Protein);
model0 = fit(LinearModel, formula, data_9);
test_direct(formula, data_9, model0)
model2 = fit(LinearModel, 
             @formula(ADWG ~ Line + Protein + Group + Line&Protein), 
             data_9)
### plot the residuals
residuals(model2) |> JarqueBeraTest
plot(x = residuals(model2), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("ADWG"))

## 1.6 dry_carcass_wt 
## model1 
formula_full = @formula(dry_carcass_wt  ~ log_worm + log_EIC);
model0 = fit(LinearModel,
             formula_full,
             data_9);
## LRT tests
test_direct(formula_full, data_9, model0)

model1 = fit(LinearModel,
             @formula(dry_carcass_wt  ~ log_worm),
             data_9)
### plot the residuals
residuals(model1) |> JarqueBeraTest
plot(x = residuals(model1), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("dry_carcass_wt"))

## model2 
formula = @formula(dry_carcass_wt ~ log_worm + log_EIC + Line + Protein + Group + Line&Protein + Line&Group + Protein&Group);
model0 = fit(LinearModel, formula, data_9);
### LRT tests
test_interactions_6(formula, data_9, model0)
formula = @formula(dry_carcass_wt ~ log_worm + log_EIC + Line + Protein + Group + Line&Protein);
model0 = fit(LinearModel, formula, data_9);
test_direct(formula, data_9, model0)
model2 = fit(LinearModel, 
             @formula(dry_carcass_wt ~ log_worm + Line + Protein + Group + Line&Protein), 
             data_9)
### plot the residuals
residuals(model2) |> JarqueBeraTest
plot(x = residuals(model2), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("dry_carcass_wt"))

# 1.7 log_EIC
formula = @formula(log_EIC ~ Line + Protein + Group + Line&Protein + Line&Group + Protein&Group);
model0 = fit(LinearModel, formula, data_9);
### LRT tests
test_interactions(formula, data_9, model0)
formula = @formula(log_EIC ~ Line + Protein + Group );
model0 = fit(LinearModel, formula, data_9);
test_direct(formula, data_9, model0)
model2 = fit(LinearModel, 
             @formula(log_EIC ~ 1), 
             data_9)

residuals(model2) |> JarqueBeraTest
plot(x = residuals(model2), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("log_EIC"))

# 1.8 log_worm
formula = @formula(log_worm ~ Line + Protein + Group + Line&Protein + Line&Group + Protein&Group);
model0 = fit(LinearModel, formula, data_9);
### LRT tests
test_interactions(formula, data_9, model0)
formula = @formula(log_worm ~ Line + Protein + Group );
model0 = fit(LinearModel, formula, data_9);
test_direct(formula, data_9, model0)
model2 = fit(LinearModel, 
             @formula(log_worm ~ Line + Group), 
             data_9)

residuals(model2) |> JarqueBeraTest
plot(x = residuals(model2), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("log_worm"))






# 2. Dpi == 21
# 2.1 Analyse _iNOS_QPCR_Gut 
## model1 
formula_full = @formula(_iNOS_QPCR_Gut  ~ log_worm + log_EIC);
model0 = fit(LinearModel,
             formula_full,
             data_21);
## LRT tests
test_direct(formula_full, data_21, model0)

model1 = fit(LinearModel,
             @formula(_iNOS_QPCR_Gut  ~ 1),
             data_21)
### plot the residuals
residuals(model1) |> JarqueBeraTest
plot(x = residuals(model1), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("_iNOS_QPCR_Gut"))

### model2 
formula_full = @formula(_iNOS_QPCR_Gut ~ log_worm + log_EIC + Line + Protein + Group + Line & Protein + Line & Group + Protein & Group);
model0 = fit(LinearModel, formula_full, data_21);
### LRT tests
test_interactions_6(formula_full, data_21, model0)
formula_full = @formula(_iNOS_QPCR_Gut ~ log_worm + log_EIC + Line + Protein + Group + Line & Group + Protein & Group);
model0 = fit(LinearModel, formula_full, data_21);
test_direct(formula_full, data_21, model0)
model2 = fit(LinearModel, 
             @formula(_iNOS_QPCR_Gut ~ Line + Protein + Group + Line & Group + Protein & Group), 
             data_21)

residuals(model2) |> JarqueBeraTest
### Plot residuals
plot(x = residuals(model2), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("_iNOS_QPCR_Gut"))



# 2.2 Analyse _iNOS_QPCR_spleen
### model1 
formula_full = @formula(_iNOS_QPCR_spleen  ~ log_worm + log_EIC);
model0 = fit(LinearModel,
             formula_full,
             data_21);
## LRT tests
test_direct(formula_full, data_21, model0)

model1 = fit(LinearModel,
             @formula(_iNOS_QPCR_spleen  ~ 1),
             data_21)
### plot the residuals
residuals(model1) |> JarqueBeraTest
plot(x = residuals(model1), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("_iNOS_QPCR_spleen"))

## model2 
formula_full = @formula(_iNOS_QPCR_spleen ~ log_worm + log_EIC + Line + Protein + Group + Line&Protein + Line&Group + Protein&Group);
model0 = fit(LinearModel, formula_full, data_21);
### LRT tests
test_interactions_6(formula_full, data_21, model0)
formula = @formula(_iNOS_QPCR_spleen ~ log_worm + log_EIC + Line + Protein + Group);
model0 = fit(LinearModel, formula, data_21);
test_direct(formula, data_21, model0)
model2 = fit(LinearModel, 
             @formula(_iNOS_QPCR_spleen ~ Protein), 
             data_21)

residuals(model2) |> JarqueBeraTest
### Plot residuals
plot(x = residuals(model2), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("_iNOS_QPCR_spleen"))

### Non-parametric test
### test Line
KruskalWallisTest(filter(row -> row.Line == "ROH", dropmissing(data_21, :_iNOS_QPCR_spleen))[:, :_iNOS_QPCR_spleen], 
                  filter(row -> row.Line == "ROL", dropmissing(data_21, :_iNOS_QPCR_spleen))[:, :_iNOS_QPCR_spleen])
### test Protein
KruskalWallisTest(filter(row -> row.Protein == "HP", dropmissing(data_21, :_iNOS_QPCR_spleen))[:, :_iNOS_QPCR_spleen], 
                  filter(row -> row.Protein == "LP", dropmissing(data_21, :_iNOS_QPCR_spleen))[:, :_iNOS_QPCR_spleen])
### test Group 
KruskalWallisTest(filter(row -> row.Group == "0_0", dropmissing(data_21, :_iNOS_QPCR_spleen))[:, :_iNOS_QPCR_spleen], 
                  filter(row -> row.Group == "0_1", dropmissing(data_21, :_iNOS_QPCR_spleen))[:, :_iNOS_QPCR_spleen], 
                  filter(row -> row.Group == "1_0", dropmissing(data_21, :_iNOS_QPCR_spleen))[:, :_iNOS_QPCR_spleen], 
                  filter(row -> row.Group == "1_1", dropmissing(data_21, :_iNOS_QPCR_spleen))[:, :_iNOS_QPCR_spleen])


# 2.3 Analyse _IgE_serum    
## model1 
formula_full = @formula(_IgE_serum  ~ log_worm + log_EIC);
model0 = fit(LinearModel,
             formula_full,
             data_21);
## LRT tests
test_direct(formula_full, data_21, model0)

model1 = fit(LinearModel,
             @formula(_IgE_serum  ~ log_worm + log_EIC),
             data_21)
### plot the residuals
residuals(model1) |> JarqueBeraTest
plot(x = residuals(model1), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("_IgE_serum"))

### model2 
data_residual = dropmissing(data_21, :_IgE_serum)
dropmissing!(data_residual, :log_EIC)
data_residual.residuals = residuals(model1)
formula = @formula(residuals ~ Line + Protein + Group + Line&Protein + Line&Group + Protein&Group);
model0 = fit(LinearModel, formula, data_residual);
test_interactions(formula, data_residual, model0)
model2 = fit(LinearModel, 
             @formula(residuals ~ Line + Protein + Group + Line&Protein + Line&Group), 
             data_residual)
### plot the residuals
residuals(model2) |> JarqueBeraTest
plot(x = residuals(model2), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("IgE"))


## 2.4 ADWG 
## model1 
formula_full = @formula(ADWG  ~ log_worm + log_EIC);
model0 = fit(LinearModel,
             formula_full,
             data_21);
## LRT tests
test_direct(formula_full, data_21, model0)

model1 = fit(LinearModel,
             @formula(ADWG  ~ 1),
             data_21)
### plot the residuals
residuals(model1) |> JarqueBeraTest
plot(x = residuals(model1), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("ADWG"))

## model2 
formula = @formula(ADWG ~ log_worm + log_EIC + Line + Protein + Group + Line&Protein + Line&Group + Protein&Group);
model0 = fit(LinearModel, formula, data_21);
### LRT tests
test_interactions_6(formula, data_21, model0)
formula = @formula(ADWG ~ log_worm + log_EIC + Line + Protein + Group + Line&Protein + Line&Group);
model0 = fit(LinearModel, formula, data_21);
test_direct(formula, data_21, model0)
model2 = fit(LinearModel, 
             @formula(ADWG ~ Line + Protein + Group + Line&Protein + Line&Group), 
             data_21)
### plot the residuals
residuals(model2) |> JarqueBeraTest
plot(x = residuals(model2), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("ADWG"))


## 2.5 dry_carcass_wt 
## model1 
formula_full = @formula(dry_carcass_wt  ~ log_worm + log_EIC);
model0 = fit(LinearModel,
             formula_full,
             data_21);
## LRT tests
test_direct(formula_full, data_21, model0)

model1 = fit(LinearModel,
             @formula(dry_carcass_wt  ~ log_EIC),
             data_21)
### plot the residuals
residuals(model1) |> JarqueBeraTest
plot(x = residuals(model1), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("dry_carcass_wt"))

## model2 
formula = @formula(dry_carcass_wt ~ log_worm + log_EIC + Line + Protein + Group + Line&Protein + Line&Group + Protein&Group);
model0 = fit(LinearModel, formula, data_21);
### LRT tests
test_interactions_6(formula, data_21, model0)
formula = @formula(dry_carcass_wt ~ log_worm + log_EIC + Line + Protein + Group + Line&Protein + Line&Group);
model0 = fit(LinearModel, formula, data_21);
test_direct(formula, data_21, model0)
model2 = fit(LinearModel, 
             @formula(dry_carcass_wt ~ log_EIC + Line + Protein + Group + Line&Protein + Line&Group), 
             data_21)
### plot the residuals
residuals(model2) |> JarqueBeraTest
plot(x = residuals(model2), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("dry_carcass_wt"))


# 2.6 log_EIC
formula = @formula(log_EIC ~ Line + Protein + Group + Line&Protein + Line&Group + Protein&Group);
model0 = fit(LinearModel, formula, data_21);
### LRT tests
test_interactions(formula, data_21, model0)
formula = @formula(log_EIC ~ Line + Protein + Group + Line&Group);
model0 = fit(LinearModel, formula, data_21);
test_direct(formula, data_21, model0)
model2 = fit(LinearModel, 
             @formula(log_EIC ~ Line + Group + Line&Group), 
             data_21)

residuals(model2) |> JarqueBeraTest
plot(x = residuals(model2), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("log_EIC"))

# 2.7 log_worm
formula = @formula(log_worm ~ Line + Protein + Group + Line&Protein + Line&Group + Protein&Group);
model0 = fit(LinearModel, formula, data_21);
### LRT tests
test_interactions(formula, data_21, model0)
formula = @formula(log_worm ~ Line + Protein + Group + Line&Group);
model0 = fit(LinearModel, formula, data_21);
test_direct(formula, data_21, model0)
model2 = fit(LinearModel, 
             @formula(log_worm ~ Line + Protein + Group + Line&Group), 
             data_21)

residuals(model2) |> JarqueBeraTest
plot(x = residuals(model2), 
     Geom.histogram(), 
     Guide.xlabel("Residuals"), 
     Guide.ylabel("Count"), 
     Guide.title("log_worm"))


# Plot
for feature in feat_sel
    p1 = Gadfly.plot(dropmissing(data_9, feature),
                     xgroup = :Line,
                     x = :Group, 
                     y = feature,
                     color = :Protein,  
                     Geom.subplot_grid(Geom.boxplot(suppress_outliers = true)),
                     Scale.color_discrete_manual("purple", "orange", order = [1, 2]),
                     Guide.xlabel(""), 
                     Guide.ylabel("$feature"), 
                     Guide.title("Dpi = 9"))
    draw(PDF(joinpath(homedir(), "Desktop/" * string(feature) * "9.pdf"), 10cm, 10cm), p1)
    p2 = Gadfly.plot(dropmissing(data_21, feature),
                     xgroup = :Line,
                     x = :Group, 
                     y = feature,
                     color = :Protein,  
                     Geom.subplot_grid(Geom.boxplot(suppress_outliers = true)),
                     Scale.color_discrete_manual("purple", "orange", order = [2, 1]),
                     Guide.xlabel(""), 
                     Guide.ylabel("$feature"), 
                     Guide.title("Dpi = 21"))
    draw(PDF(joinpath(homedir(), "Desktop/" * string(feature) * "21.pdf"), 10cm, 10cm), p2)
end

