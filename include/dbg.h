#ifndef _DBG_H_
#define _DBG_H_

#ifdef _DBG_
#include <iostream>
#define dbg_print() do {\
	std::cout << RED << "---------------------debug location " << __FILE__ << ":" << __LINE__ << " ----------------------------" << NORMAL << std::endl << std::flush;\
} while(0)
#else
#define dbg_print() do {} while(0)
#endif

#endif
