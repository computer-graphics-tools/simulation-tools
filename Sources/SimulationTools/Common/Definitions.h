#ifndef Definitions_h
#define Definitions_h

constant bool deviceSupportsNonuniformThreadgroups [[ function_constant(0) ]];
constant bool deviceDoesntSupportNonuniformThreadgroups = !deviceSupportsNonuniformThreadgroups;

#endif /* Definitions_h */

