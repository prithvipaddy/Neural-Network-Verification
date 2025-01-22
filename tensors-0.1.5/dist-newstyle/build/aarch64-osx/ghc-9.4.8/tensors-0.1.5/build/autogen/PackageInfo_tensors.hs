{-# LANGUAGE NoRebindableSyntax #-}
{-# OPTIONS_GHC -fno-warn-missing-import-lists #-}
{-# OPTIONS_GHC -w #-}
module PackageInfo_tensors (
    name,
    version,
    synopsis,
    copyright,
    homepage,
  ) where

import Data.Version (Version(..))
import Prelude

name :: String
name = "tensors"
version :: Version
version = Version [0,1,5] []

synopsis :: String
synopsis = "Tensor in Haskell"
copyright :: String
copyright = "(c) 2018 Daniel YU"
homepage :: String
homepage = "https://github.com/leptonyu/tensors#readme"
