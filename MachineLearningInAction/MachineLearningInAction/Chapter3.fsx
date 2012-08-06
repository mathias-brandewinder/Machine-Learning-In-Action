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
