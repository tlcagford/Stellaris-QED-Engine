#!/usr/bin/env python3
"""
Dual License Setup Script for Tony Eugene Ford's repositories
"""

import os
from datetime import datetime

def create_license_files():
    """Create all license files with proper attribution"""
    
    # Read the license templates we created above
    apache_license = """Copyright {year} Tony Eugene Ford

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.""".format(year=datetime.now().year)

    mit_license = """Copyright (c) {year} Tony Eugene Ford

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.""".format(year=datetime.now().year)

    dual_notice = """STELLARIS QED ENGINE
Copyright (c) {year} Tony Eugene Ford (tlcagford@gmail.com)

This project is dual-licensed under both:

1. Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
2. MIT License (https://opensource.org/licenses/MIT)

You may choose to use, modify, and distribute this software under either license.

The Apache License provides additional patent protection and explicit 
contributor license agreements, while the MIT License offers simplicity.

For commercial applications, you may prefer the Apache 2.0 license for its
explicit patent protections. For academic or personal use, the MIT license
offers maximum flexibility.

See individual license files for complete terms:
- LICENSE-APACHE (Apache 2.0)
- LICENSE-MIT (MIT)""".format(year=datetime.now().year)

    # Write files
    with open('LICENSE-APACHE', 'w') as f:
        f.write(apache_license)
    
    with open('LICENSE-MIT', 'w') as f:
        f.write(mit_license)
        
    with open('LICENSE', 'w') as f:
        f.write(dual_notice)
        
    print("✅ Dual license files created for Tony Eugene Ford")
    print("📄 LICENSE-APACHE (Apache 2.0)")
    print("📄 LICENSE-MIT (MIT)") 
    print("📄 LICENSE (Dual license notice)")

if __name__ == "__main__":
    create_license_files()
