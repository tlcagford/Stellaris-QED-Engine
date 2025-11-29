#!/usr/bin/env python3
"""
Setup Dual License for STELLARIS QED ENGINE
Commercial + Academic licensing model
"""

def create_dual_license():
    main_license = """# DUAL LICENSE - STELLARIS QED ENGINE
Copyright (c) 2024 Tony Eugene Ford (tlcagford@gmail.com)

This project is available under two distinct licenses:

## OPTION 1: COMMERCIAL LICENSE
For commercial, enterprise, or for-profit use.

Required for:
- Commercial applications
- Enterprise use  
- For-profit integration
- Proprietary derivatives

Contact: tlcagford@gmail.com for commercial licensing terms.

## OPTION 2: OPEN ACADEMIC & PERSONAL LICENSE
For academic research, personal projects, and non-commercial use.

Permissions:
- ✅ Academic research and publications
- ✅ Personal projects and experimentation  
- ✅ Non-commercial educational use
- ✅ Open source derivatives (must remain open)

Restrictions:
- ❌ No commercial use without commercial license
- ❌ No proprietary derivatives without commercial license

## SUMMARY:
- Academic/Personal: FREE (with attribution)
- Commercial: Requires paid license

See individual license files for complete terms."""
    
    academic_license = """OPEN ACADEMIC & PERSONAL LICENSE
Version 1.0, January 2024

Copyright (c) 2024 Tony Eugene Ford (tlcagford@gmail.com)

PERMITTED USES:
1. Academic research at educational institutions
2. Publication in academic journals and conferences
3. Personal projects and experimentation
4. Classroom and educational use
5. Open source derivatives (must remain open source)

CONDITIONS:
1. Attribution must be given in publications and derivatives
2. Derivatives must remain under this license or compatible open source license
3. No commercial use without separate commercial license

PROHIBITED:
1. Commercial, enterprise, or for-profit use
2. Integration into proprietary software
3. Use in products or services sold for profit

For commercial licensing, contact: tlcagford@gmail.com

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHOR BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""
    
    commercial_placeholder = """COMMERCIAL LICENSE - STELLARIS QED ENGINE

Commercial licensing is required for:
- Enterprise use
- Commercial applications
- For-profit integration  
- Proprietary derivatives

CONTACT: Tony Eugene Ford
EMAIL: tlcagford@gmail.com
SUBJECT: Commercial License Inquiry - STELLARIS QED ENGINE

Please include:
- Your organization name
- Intended use case  
- Expected scale of deployment

Contact us for custom commercial licensing terms."""
    
    with open('LICENSE', 'w') as f:
        f.write(main_license)
    
    with open('ACADEMIC_LICENSE', 'w') as f:
        f.write(academic_license)
        
    with open('COMMERCIAL_LICENSE', 'w') as f:
        f.write(commercial_placeholder)
    
    print("✅ Dual license setup complete!")
    print("📄 LICENSE - Main dual license explanation")
    print("📄 ACADEMIC_LICENSE - Free for non-commercial use") 
    print("📄 COMMERCIAL_LICENSE - Contact for commercial terms")

if __name__ == "__main__":
    create_dual_license()
