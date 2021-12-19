#include "nccl.h"

size_t size_of_type(ncclDataType_t dtype) {
  switch (dtype) {
   case ncclInt8:
   case ncclUint8:
    return 1;
   case ncclFloat16:
   case ncclBfloat16:
    return 2;
   case ncclInt32:
   case ncclUint32:
   case ncclFloat32:
    return 4;
   case ncclInt64:
   case ncclUint64:
   case ncclFloat64:
    return 8;
  }
  return 0;
}
