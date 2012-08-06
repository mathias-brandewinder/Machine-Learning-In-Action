#load "DecisionTrees.fs"
open MachineLearning.DecisionTrees

// Construct a Decision Tree by hand
let manualTree = 
    Choice
        ("Sci-Fi",
         [|("No",
            Choice
              ("Action",
               [|("Yes", Conclusion "Stallone");
                 ("No", Conclusion "Schwarzenegger")|]));
           ("Yes", Conclusion "Schwarzenegger")|])

// Use the tree to Classify a test Subject
let test = [| ("Action", "Yes"); ("Sci-Fi", "Yes") |]
let actor = classify test manualTree

// Sample dataset
let movies =
    [| "Action"; "Sci-Fi"; "Actor" |],
    [| [| "Yes"; "No";  "Stallone" |];
       [| "Yes"; "No";  "Stallone" |];
       [| "No";  "No";  "Schwarzenegger"  |];
       [| "Yes"; "Yes"; "Schwarzenegger"  |];
       [| "Yes"; "Yes"; "Schwarzenegger"  |] |]

// Construct the Decision Tree off the data
// and classify another test subject
let tree = build movies
let subject = [| ("Action", "Yes"); ("Sci-Fi", "No") |]
let answer = classify subject tree

// Lenses dataset: http://archive.ics.uci.edu/ml/datasets/Lenses
// The following code assumes the dataset has been massaged a bit,
// and that data has been transformed to be comma-separated instead of tab.

//open System.IO
//let lenses = 
//    let file = the path to the file goes here
//    let fileAsLines =
//        File.ReadAllLines(file)
//        |> Array.map (fun line -> line.Split(','))
//    let dataset = 
//        fileAsLines
//        |> Array.map (fun line -> 
//            [| line.[0]
//               line.[1]; 
//               line.[2]; 
//               line.[3];
//               line.[4]|])
//    let labels = [| "Age"; "Presc."; "Astigm"; "Tears"; "Decision" |]
//    labels, dataset

// Nursery Dataset: http://archive.ics.uci.edu/ml/datasets/Nursery

//open System.IO
//let nursery =
//    let file = the path to the file goes here
//    let fileAsLines =
//        File.ReadAllLines(file)
//        |> Array.map (fun line -> line.Split(','))
//    let labels = [| "parents"; "has_nurs"; "form"; "children"; "housing"; "finance"; "social"; "health"; "Decision" |]
//    let dataset = 
//        fileAsLines
//        |> Array.map (fun line -> 
//            [| line.[0]
//               line.[1]; 
//               line.[2]; 
//               line.[3];
//               line.[4]
//               line.[5]
//               line.[6]
//               line.[7]
//               line.[8] |])
//    labels, dataset