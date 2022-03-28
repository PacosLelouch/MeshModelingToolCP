// This file is part of libigl, a simple c++ geometry processing library.
// 
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef IGL_SORTABLE_ROW_H
#define IGL_SORTABLE_ROW_H

// Simple class to contain a rowvector which allows rowwise sorting and
// reordering
#include <Eigen/Core>

namespace igl
{
  // Templates:
  //   T  should be a matrix that implments .size(), and operator(int i)
  template <typename T>
  class SortableRow
  {
    public:
      T data;
    public:
      SortableRow():data(){};
      SortableRow(const T & data):data(data){};
      bool operator<(const SortableRow & that) const
      {
        // Get reference so that I can use parenthesis
        const SortableRow<T> & thisRef = *this;
        // Lexicographical
        int minc = (thisRef.data.size() < that.data.size()? 
            thisRef.data.size() : that.data.size());
        // loop over columns
        for(int i = 0;i<minc;i++)
        {
          if(thisRef.data(i) == that.data(i))
          {
            continue;
          }
          return thisRef.data(i) < that.data(i);
        }
        // All characters the same, comes done to length
        return thisRef.data.size()<that.data.size();
      };
      bool operator==(const SortableRow & that) const
      {
        // Get reference so that I can use parenthesis
        const SortableRow<T> & thisRef = *this;
        if(thisRef.data.size() != that.data.size())
        {
          return false;
        }
        for(int i = 0;i<thisRef.data.size();i++)
        {
          if(thisRef.data(i) != that.data(i))
          {
            return false;
          }
        }
        return true;
      };
      bool operator!=(const SortableRow & that) const
      {
        return !(*this == that);
      };
  };
}

#endif
