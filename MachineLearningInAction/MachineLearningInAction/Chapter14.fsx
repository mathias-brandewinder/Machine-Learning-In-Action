// Chapter 14 covers SVD (Singular Value Decomposition)

#r @"..\..\MachineLearningInAction\packages\MathNet.Numerics.2.5.0\lib\net40\MathNet.Numerics.dll"
#r @"..\..\MachineLearningInAction\packages\MathNet.Numerics.FSharp.2.5.0\lib\net40\MathNet.Numerics.FSharp.dll"

open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Statistics

type Rating = { UserId: int; DishId: int; Rating: int }

// Our existing "ratings database"
let ratings = [
    { UserId = 0; DishId = 0; Rating = 2 };
    { UserId = 0; DishId = 3; Rating = 4 };
    { UserId = 0; DishId = 4; Rating = 4 };
    { UserId = 1; DishId = 10; Rating = 5 };
    { UserId = 2; DishId = 7; Rating = 1 };
    { UserId = 2; DishId = 0; Rating = 4 };
    { UserId = 3; DishId = 0; Rating = 3 };
    { UserId = 3; DishId = 1; Rating = 3 };
    { UserId = 3; DishId = 2; Rating = 4 };
    { UserId = 3; DishId = 4; Rating = 3 };
    { UserId = 3; DishId = 7; Rating = 2 };
    { UserId = 3; DishId = 8; Rating = 2 };
    { UserId = 4; DishId = 0; Rating = 5 };
    { UserId = 4; DishId = 1; Rating = 5 };
    { UserId = 4; DishId = 2; Rating = 5 };
    { UserId = 5; DishId = 6; Rating = 5 };
    { UserId = 5; DishId = 9; Rating = 5 };
    { UserId = 6; DishId = 0; Rating = 4 };
    { UserId = 6; DishId = 2; Rating = 4 };
    { UserId = 6; DishId = 10; Rating = 5 };
    { UserId = 7; DishId = 5; Rating = 4 };
    { UserId = 7; DishId = 10; Rating = 4 };
    { UserId = 8; DishId = 6; Rating = 5 };
    { UserId = 9; DishId = 3; Rating = 3 };
    { UserId = 9; DishId = 8; Rating = 4 };
    { UserId = 9; DishId = 9; Rating = 5 };
    { UserId = 10; DishId = 0; Rating = 1 };
    { UserId = 10; DishId = 1; Rating = 1 };
    { UserId = 10; DishId = 2; Rating = 2 };
    { UserId = 10; DishId = 3; Rating = 1 };
    { UserId = 10; DishId = 4; Rating = 1 };
    { UserId = 10; DishId = 5; Rating = 2 };
    { UserId = 10; DishId = 6; Rating = 1 };
    { UserId = 10; DishId = 8; Rating = 4 };
    { UserId = 10; DishId = 9; Rating = 5 } ]

// Let's populate a matrix with these ratings;
// unrated items are denoted by a 0

let rows = 11
let cols = 11
let data = DenseMatrix(rows, cols)
ratings 
|> List.iter (fun rating -> 
       data.[rating.UserId, rating.DishId] <- (float)rating.Rating)

// "Pretty" rendering of a matrix
// something is not 100% right here, but it's good enough
let printNumber v = 
    if v < 0. 
    then printf "%.2f " v 
    else printf " %.2f " v
// Display a Matrix in a "pretty" format
let pretty matrix = 
    Matrix.iteri (fun row col value ->
        if col = 0 then printfn "" else ignore ()
        printNumber value) matrix
    printfn ""

printfn "Original data matrix"
pretty data

// Now let's run a SVD on that matrix
let svd = data.Svd(true)
let U, sigmas, Vt = svd.U(), svd.S(), svd.VT()

// recompose the S matrix from the singular values
printfn "S-matrix, with singular values in diagonal"
let S = DiagonalMatrix(rows, cols, sigmas.ToArray())
pretty S

// The SVD decomposition verifies
// data = U x S x Vt

printfn "Reconstructed matrix from SVD decomposition"
let reconstructed = U * S * Vt
pretty reconstructed

// Can we interpret the SVD decomposition? Let's see.

// Each row maps to a User, 
// each column to an extracted Category
let userToCategory = U * S |> pretty

// Each row maps to an extracted Category, 
// each column to a Dish
let categoryToDish = S * Vt |> pretty

// We can use SVD as a data compression mechanism
// by retaining only the n first singular values

// Total energy is the sum of the squares 
// of the singular values
let totalEnergy = sigmas.DotProduct(sigmas)
// Let's compute how much energy is contributed
// by each singular value
printfn "Energy contribution by Singular Value"
sigmas.ToArray() 
|> Array.fold (fun acc x ->
       let energy = x * x
       let percent = (acc + energy)/totalEnergy
       printfn "Energy: %.1f, Percent of total: %.3f" energy percent
       acc + energy) 0. 
|> ignore

// We'll retain only the first 5 singular values,
// which cover 90% of the total energy
let subset = 5
let U' = U.SubMatrix(0, U.RowCount, 0, subset)
let S' = S.SubMatrix(0, subset, 0, subset)
let Vt' = Vt.SubMatrix(0, subset, 0, Vt.ColumnCount)

// Using U', S', Vt' instead of U, S, Vt
// should produce a "decent" approximation
// of the original matrix

printfn "Approximation of the original matrix"
U' * S' * Vt' |> pretty