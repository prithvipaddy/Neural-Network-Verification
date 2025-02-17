{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE FlexibleContexts #-}
-- {-# LANGUAGE LiquidHaskell #-}

module Main where

import Data.List (maximumBy,transpose)
import Data.Aeson (FromJSON, decode, encode)
import qualified Data.ByteString.Lazy as B (readFile, writeFile)
import GHC.Generics (Generic)
import Data.Ord (comparing)
import Data.Vector (toList)
import Data.Matrix
import Data.Maybe (fromMaybe)
import System.Environment (getArgs)
import Control.Monad (liftM2)

type Zonotope = Matrix Double

data LayerInfo = LayerInfo
  { layerName          :: String
  , layerType          :: String
  , inputShape         :: Maybe (Int,Int)
  , numFilters         :: Maybe Int
  , kernelSize         :: Maybe [Int]
  , activationFunction :: Maybe String
  , filters            :: Maybe [[[[Double]]]]
  , biases             :: Maybe [Double]
  , poolSize           :: Maybe [Int]
  , units              :: Maybe Int
  , weights            :: Maybe [[Double]]
  } deriving (Show, Generic)
instance FromJSON LayerInfo

data ImageData = ImageData
  { imageValues          :: [[[Double]]]
  , imageClass           :: Int
  , imageDimensions      :: (Int,Int,Int)
  } deriving (Show, Generic)
instance FromJSON ImageData

-- CONVOLUTION (from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8418593)

-- Function to generate one row of the weight matrix wF
generateRowWeightMatrixForConv :: [[Double]] -> (Int,Int) -> Int -> Int -> [Double]
generateRowWeightMatrixForConv kernel (inpImgRows,inpImgCols) i j =
  let kernelRows = length kernel
      kernelCols = length (head kernel)
      -- The if condition checks for a box the size of the kernel
      -- The position of the box is determined by i and j
      -- if i' and j' belong to the box then the relevent element from the kernel is selected
      row = [ if i' >= i && i' < i + kernelRows && j' >= j && j' < j + kernelCols
                then kernel !! (i' - i) !! (j' - j)
                else 0
            | i' <- [0 .. inpImgRows - 1]
            , j' <- [0 .. inpImgCols - 1]
            ]
   in row

-- Function to generate the full weight matrix wF
generateWeightMatrixForConv :: [[Double]] -> (Int,Int) -> [[Double]]
generateWeightMatrixForConv kernel (inpImgRows,inpImgCols) =
  let kernelRows = length kernel
      kernelCols = length (head kernel)
    -- Assuming that there is no padding and the stride is 1, we get the y dimensions below
    -- Here y signifies the output from convolution between the kernel and the input image
      yRows = inpImgRows - kernelRows + 1
      yCols = inpImgCols - kernelCols + 1
    -- One row of the Wf matrix transforms into one element of the y matrix
    -- i and j determine the row corresponding to element (i,j) from y
   in [ generateRowWeightMatrixForConv kernel (inpImgRows,inpImgCols) i j | i <- [0 .. yRows - 1], j <- [0 .. yCols - 1] ]

generateBiasMatrixForConv :: Matrix Double -> Double -> Matrix Double
generateBiasMatrixForConv z b = matrix (nrows z) (ncols z) (const b)

applySingleChannelConvolution :: Matrix Double -> (Int,Int) -> [[Double]] -> Matrix Double
applySingleChannelConvolution zonotope (inpImgRows,inpImgCols) kernel = let
    wF = Data.Matrix.fromLists (generateWeightMatrixForConv kernel (inpImgRows,inpImgCols))
    newZonotope = Data.Matrix.multStd wF zonotope
    in newZonotope

-- scalar composotion of the entire depth of the zonotopes to form a single zonotope
-- one scalar composition for each filter 
-- eg. if initial depth = 3, newZ is sum of the 3 zonotopes, and if numFilters = 32, new depth = 32
applyConvolutionPerFilter' :: [Matrix Double] -> (Int,Int) -> [[[Double]]] -> Int -> Int -> Matrix Double
applyConvolutionPerFilter' [] _ _ zRowSize zColSize = Data.Matrix.zero zRowSize zColSize
applyConvolutionPerFilter' _ _ [] zRowSize zColSize = Data.Matrix.zero zRowSize zColSize
applyConvolutionPerFilter' (z:zonotope) (inpImgRows,inpImgCols) (k:kernel) zRowSize zColSize  = let
    newZ = applySingleChannelConvolution z (inpImgRows,inpImgCols) k
    in newZ + applyConvolutionPerFilter' zonotope (inpImgRows,inpImgCols) kernel zRowSize zColSize

applyConvolutionPerFilter :: [Matrix Double] -> (Int,Int) -> [[[Double]]] -> Double -> Matrix Double
applyConvolutionPerFilter zonotope (inpImgRows,inpImgCols) kernel bias = let
    convolved = applyConvolutionPerFilter' zonotope (inpImgRows,inpImgCols) kernel (nrows (head zonotope)) (ncols (head zonotope))
    biasMatrix = generateBiasMatrixForConv convolved bias
    final = convolved + biasMatrix
    in final

applyConvolution :: [Matrix Double] -> (Int,Int) -> [[[[Double]]]] -> [Double] -> [Matrix Double]
applyConvolution _ _ [] _ = []
applyConvolution _ _ _ [] = []
-- length of kernel and bias should be same
applyConvolution zonotope (inpImgRows,inpImgCols) (k:kernel) (b:bias) = applyConvolutionPerFilter zonotope (inpImgRows,inpImgCols) k b : applyConvolution zonotope (inpImgRows,inpImgCols) kernel bias

-- MAXPOOLING (from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8418593)
-- Generate the list of indices grouped for maxpooling
poolingGroups :: Int -> Int -> [[Double]] -> [[Int]]
poolingGroups p q x =
  let rows = length x
      cols = length (head x)
      -- groupIndices finds the indices for all elements in a pooling box.
      -- The input to the function is the top-left element of each box
      groupIndices i j = [ (i + di) * cols + (j + dj) | di <- [0 .. p - 1], dj <- [0 .. q - 1] ]
      -- Below, the groupIndices for the top-left element of every pooling group is found
   in [ groupIndices (i * p) (j * q) | i <- [0 .. (rows `div` p) - 1], j <- [0 .. (cols `div` q) - 1] ]

appendColumn :: Matrix a -> [a] -> Matrix a
appendColumn mat col
  | nrows mat /= length col = error "Column length must match the number of rows in the matrix"
  | otherwise =
      let
        matRows = toLists mat
        newMatRows = zipWith (++) (map (:[]) col) matRows  -- Append each element of the column to each row
      in fromLists newMatRows

generateWMPPooling :: Int -> Int -> [[Double]] -> [[Double]]
generateWMPPooling p q x =
  let
      xv = concat x
      -- groupMap is a list of indices, which is used to reorder elements in xv
      groupMap = concat (poolingGroups p q x)
      sizeX = length xv
      sizeG = length groupMap
      -- Below, the i determines which row of wMP is being generated
      -- j determines which position of the row will be occupied by '1'
   in [[if i >= sizeG then 0.0 else if groupMap !! i == j then 1.0 else 0.0 | j <- [0 .. sizeX - 1]] | i <- [0 .. sizeX - 1]]

{-
Example usage of generateWPooling
p = 2, q = 1
x:
[0,1,2,3]
[4,5,6,7]
[8,9,10,11]
[12,13,14,15]
xv = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

groups: [[0,4],[1,5],[2,6],[3,7],[8,12],[9,13],[10,14],[11,15]] 
NOTE FOR GROUPS: these will always be indices, it just so happens that x has been defined with values matching indices
groupMap = [0,4,1,5,2,6,3,7,8,12,9,13,10,14,11,15]

when i = 0, checks the value for groupMap at index 0.
since it is 0, it will input 1 into the first position of row 0.

i = 1, checks the value for groupMax at index 1.
passes through all j values upto 4, inserting '0' in those positions. at index 4, it will insert a '1'
this indicates that the 5th element from xv should be the 2nd element in xMP

Similarly done, upto i = 15, and wMP is constructed.
-}

maxPooling :: Int -> Int -> [[Double]] -> [Double]
maxPooling p q x = let
    rows = length x
    cols = length (head x)
    newRows = rows `div` p
    newCols = cols `div` q
    wMP = Data.Matrix.transpose (Data.Matrix.fromLists (generateWMPPooling p q x))
    xFlat = Data.Matrix.fromLists [concat x]
    xMP' = multStd xFlat wMP
    xMP = Data.Matrix.toList xMP'
    sizeIdentity = length xMP
    size = length (concat (poolingGroups p q x))
    identity' = [[if i == j then 1 else 0 | j <- [0..sizeIdentity-1]] | i <- [0..sizeIdentity-1]]
    sublists = [take (p*q) (drop (i * p * q) xMP) | i <- [0..size `div` (p*q) - 1]]
    maxIndices = [startIdx + fst (maximumBy (comparing snd) (zip [0..] sublist)) | (startIdx, sublist) <- zip [0, p * q ..] sublists]
    selectedRows = Data.Matrix.fromLists [row | (idx, row) <- zip [0..] identity', idx `elem` maxIndices]
    maxPooledVector = Data.Matrix.toList (multStd xMP' (Data.Matrix.transpose selectedRows))
    maxPooledMatrix = Data.Matrix.toList (Data.Matrix.fromList newRows newCols maxPooledVector)
    in
        maxPooledMatrix

applySinglePointMaxPooling :: Int -> Int -> [Double] -> (Int,Int) -> [Double]
applySinglePointMaxPooling p q zonotope (rows,cols) = let
  x = Data.Matrix.toLists (Data.Matrix.fromList rows cols zonotope)
  in maxPooling p q x

applyAllPointsMaxPooling :: Int -> Int -> Matrix Double -> Int -> (Int,Int) -> Matrix Double
applyAllPointsMaxPooling p q _ 0 (imgRows,imgCols) = matrix ((imgRows `div` p)*(imgCols `div` q)) 0 (const 0)
applyAllPointsMaxPooling p q zonotope numPoints (imgRows,imgCols) = let
  point = Data.Vector.toList (getCol numPoints zonotope)
  maxPooledPoint = applySinglePointMaxPooling p q point (imgRows,imgCols)
  in appendColumn (applyAllPointsMaxPooling p q zonotope (numPoints - 1) (imgRows,imgCols)) maxPooledPoint

applyMaxPooling :: Int -> Int -> [Matrix Double] -> (Int,Int) -> [Matrix Double]
applyMaxPooling _ _ [] _ = []
applyMaxPooling p q (z:zonotope) (imgRows,imgCols) =  applyAllPointsMaxPooling p q z (ncols z) (imgRows,imgCols) : applyMaxPooling p q zonotope (imgRows,imgCols)

-- AVERAGE POOLING
generateRowAveragePooling :: [Int] -> Int -> [Double]
generateRowAveragePooling positions size = [if i `elem` positions then 1.0 else 0.0 | i <- [0..size-1]]

averagePooling :: Int -> Int -> [[Double]] -> [Double]
averagePooling p q x = let
  groups = poolingGroups p q x
  sizeX = length x * length (head x)
  poolSize' = fromIntegral p * fromIntegral q :: Double
  w = Data.Matrix.fromLists [generateRowAveragePooling g sizeX | g <- groups]
  zonotope = Data.Matrix.transpose (Data.Matrix.fromLists [concat x])
  in Data.Matrix.toList (Data.Matrix.scaleMatrix (1/poolSize') (multStd w zonotope))

applySinglePointAveragePooling :: Int -> Int -> [Double] -> (Int,Int) -> [Double]
applySinglePointAveragePooling p q zonotope (rows,cols) = let
  x = Data.Matrix.toLists (Data.Matrix.fromList rows cols zonotope)
  in averagePooling p q x

applyAllPointsAveragePooling :: Int -> Int -> Matrix Double -> Int -> (Int,Int) -> Matrix Double
applyAllPointsAveragePooling p q _ 0 (imgRows,imgCols) = matrix ((imgRows `div` p)*(imgCols `div` q)) 0 (const 0)
applyAllPointsAveragePooling p q zonotope numPoints (imgRows,imgCols) = let
  point = Data.Vector.toList (getCol numPoints zonotope)
  avgPooledPoint = applySinglePointAveragePooling p q point (imgRows,imgCols)
  in appendColumn (applyAllPointsMaxPooling p q zonotope (numPoints - 1) (imgRows,imgCols)) avgPooledPoint

applyAveragePooling :: Int -> Int -> [Matrix Double] -> (Int,Int) -> [Matrix Double]
applyAveragePooling _ _ [] _ = []
applyAveragePooling p q (z:zonotope) (imgRows,imgCols) =  applyAllPointsAveragePooling p q z (ncols z) (imgRows,imgCols) : applyAveragePooling p q zonotope (imgRows,imgCols)

-- RELU
singleReluValue :: Double -> Double
singleReluValue x | x > 0 = x
                  | otherwise = 0

applyRelu :: [[[Double]]] -> [[[Double]]]
applyRelu = map (map (map singleReluValue))

-- RELU (SHEARING)
getZonotopeCenter :: Zonotope -> [Double]
getZonotopeCenter z = Data.Vector.toList (getCol 1 z)

getZonotopeGenerators :: Zonotope -> Zonotope
getZonotopeGenerators z = submatrix 1 (nrows z) 2 (ncols z) z

buildMatrix :: [Double] -> Zonotope -> Zonotope
buildMatrix center generators = fromLists (zipWith (:) center (toLists generators))

upper :: Zonotope -> [Double]
upper m = zipWith (+) c (map (sum . map abs) (Data.List.transpose g))
  where
    c = getZonotopeCenter m
    g = toLists (getZonotopeGenerators m)

lower :: Zonotope -> [Double]
lower m = zipWith (-) c (map (sum . map abs) (Data.List.transpose g))
  where
    c = getZonotopeCenter m
    g = toLists (getZonotopeGenerators m)

findLambdaAndNRelu :: Zonotope -> ([Double], [Double])
findLambdaAndNRelu m = (lambda, n)
  where
    u = upper m
    l = lower m
    lambda = zipWith (/) u (zipWith (-) u l)
    n = zipWith (\ui lambdai -> (ui * (1 - lambdai)) / 2) u lambda

scaleByLambdaRelu :: Zonotope -> Zonotope
scaleByLambdaRelu m = buildMatrix scaledC (fromLists scaledGs)
  where
    (lambda, _) = findLambdaAndNRelu m
    c = getZonotopeCenter m
    g = toLists (getZonotopeGenerators m)
    scaledC = zipWith (*) lambda c
    scaledGs = map (zipWith (*) lambda) g

composeLambdaAndNRelu :: Zonotope -> Zonotope
composeLambdaAndNRelu m = elementwise (+) lambdaScaled (Data.Matrix.fromLists [nRow])
  where
    lambdaScaled = scaleByLambdaRelu m
    (_, n) = findLambdaAndNRelu m
    nRow = n ++ replicate (ncols m - 1) 0 ++ [head n]

applyReluShearing :: Zonotope -> Zonotope
applyReluShearing m = fromLists (map (Data.Matrix.toList . composeLambdaAndNRelu . fromLists . pure) (toLists m))

-- DENSE
-- weights x zonotope
applyDenseLayerWeights :: Matrix Double -> Matrix Double -> Matrix Double
applyDenseLayerWeights = multStd

--TESTING APPLYING CONV AND MAXPOOLING TO A TEST ZONOTOPE
--  dimensions = 3 x 16 x 5
testZ1 :: [Matrix Double]
testZ1 = replicate 3 (Data.Matrix.fromLists (replicate 16 [1,2,-3,4,-5]))

-- dimensions = 3 x 1849 x 4
testZ2 :: [Matrix Double]
testZ2 = replicate 3 (Data.Matrix.fromLists (replicate 1849 [0.8,0.9,1,1,1.1]))

readLayers :: String -> IO [LayerInfo]
readLayers filepath = do
  jsonData <- B.readFile filepath
  let layers = fromMaybe [] (decode jsonData :: Maybe [LayerInfo])
  return layers

processLayers :: String -> [Matrix Double] -> IO [Matrix Double]
processLayers filepath zonotope = do
  layers <- readLayers filepath
  case layers of
    [] -> do
      putStrLn "Layers not found"
      return []
    layers' -> do
      let inputShape' = fromMaybe (-1,-1) (inputShape (head layers'))
      case inputShape' of
        (-1,-1) -> do
          putStrLn "Input shape for the neural network not found"
          return []
        inputShape'' -> do
          finalZonotope <- parseLayers (tail layers') zonotope inputShape''
          putStrLn $ "Final zonotope dimensions: " ++ show (length finalZonotope,nrows (head finalZonotope),ncols (head finalZonotope))
          print finalZonotope
          return finalZonotope

generateZonotopeFromImage :: Double -> String -> IO ([Matrix Double],Int)
generateZonotopeFromImage perturbation filepath = do
  jsonData <- B.readFile filepath
  let image = fromMaybe [] (decode jsonData :: Maybe [ImageData])
  case image of
    [] -> do
      putStrLn "Image data not found"
      return ([],-1)
    images' -> do
      let
        img1Class = imageClass (head images')
        img1 = imageValues (head images')
        (height,width,_) = imageDimensions (head images')
        center = map (Data.Matrix.fromList (height*width) 1) (Data.List.transpose (concat img1))
        -- creating with 1 unique generator per dimension (explained below)
        numGenerators = length (head center)
        identity' = Data.Matrix.fromLists [[if i == j then perturbation else 0 | j <- [0..numGenerators-1]] | i <- [0..numGenerators-1]]
        zonotope = map (<|> identity') center

      return (zonotope,img1Class)

{-
1 unique generator per dimension:
If zonotope is 3D and its center is (2,3,0.5), then there will be generators e1, e2 and e3 acting as follows:
2 + 1 e1
3 +       1 e2
0.5 +           1 e3
i.e
[[2,1,0,0],
 [3,0,1,0],
 [0.5,0,0,1]]
-}
convertImageDataToZonotope :: String -> IO ([Matrix Double],Int)
convertImageDataToZonotope filepath = do
  jsonData <- B.readFile filepath
  let image = fromMaybe [] (decode jsonData :: Maybe [ImageData])
  case image of
    [] -> do
      putStrLn "Image data not found"
      return ([],-1)
    images' -> do
      let
        img1 = imageValues (head images')
        (height1,width1,_) = imageDimensions (head images')
        concatImg = Data.List.transpose (concat img1)
        img1Zonotope = map (Data.Matrix.fromList (height1 * width1) 1) concatImg
      let img1Class = imageClass (head images')
      return (img1Zonotope,img1Class)

parseLayers :: [LayerInfo] -> [Matrix Double] -> (Int,Int) -> IO [Matrix Double]
parseLayers [] zonotope _ = return zonotope
parseLayers (l:layers) zonotope (imgRows,imgCols) = do
  -- Print the layer type (name) of the current layer
  putStrLn $ "parsed layer: " ++ layerName l
  putStrLn $ "img dimensions: " ++ show (imgRows,imgCols)
  putStrLn $ "zonotope dimensions: " ++ show (length zonotope,nrows (head zonotope),ncols (head zonotope))
  case layerType l of
    "<class 'keras.src.layers.convolutional.conv2d.Conv2D'>" ->
      let
        kernelSize' = fromMaybe [] (kernelSize l)
        newRows = imgRows - head kernelSize' + 1
        newCols = imgCols - head (tail kernelSize') + 1
        newZ' = applyConvolution zonotope (imgRows,imgCols) (fromMaybe [] (filters l)) (fromMaybe [] (biases l))
        activation = fromMaybe [] (activationFunction l)
        newZ = if activation == "relu"
          then map Data.Matrix.fromLists (applyRelu (map Data.Matrix.toLists newZ'))
          else newZ'
      in parseLayers layers newZ (newRows,newCols)
    "<class 'keras.src.layers.pooling.max_pooling2d.MaxPooling2D'>" ->
      let poolSize' = fromMaybe [] (poolSize l)
          p = head poolSize'
          q = head (tail poolSize')
          newRows = imgRows `div` p
          newCols = imgCols `div` q
      in parseLayers layers (applyMaxPooling p q zonotope (imgRows,imgCols)) (newRows,newCols)
    "<class 'keras.src.layers.pooling.average_pooling2d.AveragePooling2D'>" ->
      let poolSize' = fromMaybe [] (poolSize l)
          p = head poolSize'
          q = head (tail poolSize')
          newRows = imgRows `div` p
          newCols = imgCols `div` q
      in parseLayers layers (applyAveragePooling p q zonotope (imgRows,imgCols)) (newRows,newCols)
    "<class 'keras.src.layers.reshaping.flatten.Flatten'>" ->
      let
        newRows = 1
        newCols = imgRows * imgCols
      in parseLayers layers [Data.Matrix.fromList (length zonotope) (ncols (head zonotope)) (concatMap Data.Matrix.toList zonotope)] (newRows,newCols)
    "<class 'keras.src.layers.core.dense.Dense'>" ->
      let weights' = fromMaybe [] (weights l)
          weightsMatrix = Data.Matrix.transpose (Data.Matrix.fromLists weights')
          zonotopeMatrix = head zonotope
          newZ' = [applyDenseLayerWeights weightsMatrix zonotopeMatrix]
          biases' = fromMaybe [] (biases l)
          biasMatrix = Data.Matrix.transpose (fromLists (replicate (ncols (head newZ')) biases'))
          newZ'' = head newZ' + biasMatrix
          activationFunction' = fromMaybe [] (activationFunction l)
          newZ = if activationFunction' == "relu"
            then map Data.Matrix.fromLists (applyRelu [Data.Matrix.toLists newZ''])
            else [newZ'']
      in parseLayers layers newZ (imgRows,imgCols)
    "<class 'keras.src.layers.regularization.dropout.Dropout'>" ->
      parseLayers layers zonotope (imgRows,imgCols)
    _ ->
      return []

-- CHECKING IF THE ARGMAX VALUE IN ALL POINTS MATCHES
-- Function to find the index of the maximum element in a column
maxIndexInColumn :: (Eq a, Ord a) => Matrix a -> Int -> Int
maxIndexInColumn m colIndex =
  let column = getCol colIndex m
      maxVal = maximum column
      rowIndex = fst . head $ filter ((== maxVal) . snd) (zip [1..] (Data.Vector.toList column))
  in rowIndex

checkListMatch :: [Int] -> Int -> Bool
checkListMatch [] _ = True
checkListMatch (l:ls) label = (l == label) && checkListMatch ls label

-- Function to check if the maximum index in all columns matches
checkMaxIndicesMatch :: Ord a => Matrix a -> Int -> Bool
checkMaxIndicesMatch m label =
  let numCols = ncols m
      indices = map (maxIndexInColumn m) [1..numCols]  -- Get max indices for all columns
  in checkListMatch indices label

-- ADDING PERTURBATIONS TO ALL DIMENSIONS (CREATING 2^N POINTS, WHERE N IS THE NUMBER OF DIMENSIONS IN THE ZONOTOPE)
cartesianProduct :: [[a]] -> [[a]]
cartesianProduct = foldr (liftM2 (:)) [[]]

-- Constraint: Input zonotope should be only 1 column
addPerturbations :: Double -> Matrix Double -> Matrix Double
addPerturbations perturbation zonotope = let
  zonotopeList = Data.Matrix.toList zonotope
  cornerPoints' = map (\x -> [x - perturbation, x + perturbation]) zonotopeList :: [[Double]]
  cornerPoints = Data.Matrix.fromLists (cartesianProduct cornerPoints')
  in Data.Matrix.transpose cornerPoints

main :: IO ([Matrix Double],[Bool])
main = do
  args <- getArgs
  let perturbation = read (head args) :: Double
  (testZ,testZlabel) <- convertImageDataToZonotope "/Users/prithvi/Documents/Krea/Capstone/AbstractVerification/Zonotope/haskell/app/imageData.json"
  let
    perturbedZonotope = map (addPerturbations perturbation) testZ
  finalZonotope <- processLayers "/Users/prithvi/Documents/Krea/Capstone/AbstractVerification/Zonotope/haskell/app/layersInfo.json" perturbedZonotope
  let correctlyClassified = map (`checkMaxIndicesMatch` (testZlabel + 1)) finalZonotope
  print correctlyClassified
  let jsonData = encode correctlyClassified
  B.writeFile "correctlyClassified.json" jsonData
  putStrLn "Data written to correctlyClassified.json"
  return (finalZonotope,correctlyClassified)