#!/usr/bin/env python3
'''
This script generates SLEIGH code for the Telink TC32 architecture.
It parses assembler format strings (derived from objdump) to determine
bitfields and operands, and generates corresponding SLEIGH definitions.

Improvements:
- Supports both 16-bit and 32-bit instructions
- Autogenerates basic P-code templates
- Restores original logic for register mapping and computed values
'''

import re
import sys

# Instruction definitions: (size, value, mask, assembler_format)
insns = [
  (16, 0x46c0, 0xffff, 'tnop%c\\t\\t\\t; (mov r8, r8)'),
  (16, 0x0000, 0xffc0, 'tand%C\\t%0-2r, %3-5r'),
  (16, 0x0040, 0xffc0, 'txor%C\\t%0-2r, %3-5r'),
  (16, 0x0080, 0xffc0, 'tshftl%C\\t%0-2r, %3-5r'),
  (16, 0x00c0, 0xffc0, 'tshftr%C\\t%0-2r, %3-5r'),
  (16, 0x0100, 0xffc0, 'tasr%C\\t%0-2r, %3-5r'),
  (16, 0x0140, 0xffc0, 'taddc%C\\t%0-2r, %3-5r'),
  (16, 0x0180, 0xffc0, 'tsubc%C\\t%0-2r, %3-5r'),
  (16, 0x01c0, 0xffc0, 'trotr%C\\t%0-2r, %3-5r'),
  (16, 0x0200, 0xffc0, 'tnand%c\\t%0-2r, %3-5r'),
  (16, 0x0240, 0xffc0, 'tneg%C\\t%0-2r, %3-5r'),
  (16, 0x0280, 0xffc0, 'tcmp%c\\t%0-2r, %3-5r'),
  (16, 0x02c0, 0xffc0, 'tcmpn%c\\t%0-2r, %3-5r'),
  (16, 0x0300, 0xffc0, 'tor%C\\t%0-2r, %3-5r'),
  (16, 0x0340, 0xffc0, 'tmul%C\\t%0-2r, %3-5r'),
  (16, 0x0380, 0xffc0, 'tbclr%C\\t%0-2r, %3-5r'),
  (16, 0x03c0, 0xffc0, 'tmovn%C\\t%0-2r, %3-5r'),
  (16, 0x6bc0, 0xfff8, 'tmcsr%c\\t%0-2r'),
  (16, 0x6bc8, 0xfff8, 'tmrcs%c\\t%0-2r'),
  (16, 0x6bd0, 0xfff8, 'tmssr%c\\t%0-2r'),
  (16, 0x6bd8, 0xfff8, 'tmrss%c\\t%0-2r'),
  (16, 0x6800, 0xfe00, 'treti\\t%O'),
  (16, 0x6000, 0xff80, 'tadd%c\\tsp, #%0-6W'),
  (16, 0x6080, 0xff80, 'tsub%c\\tsp, #%0-6W'),
  (16, 0x0700, 0xff80, 'tjex%c\\t%S%x'),
  (16, 0x0400, 0xff00, 'tadd%c\\t%D, %S'),
  (16, 0x0500, 0xff00, 'tcmp%c\\t%D, %S'),
  (16, 0x0600, 0xff00, 'tmov%c\\t%D, %S'),
  (16, 0x6400, 0xfe00, 'tpush%c\\t%N'),
  (16, 0x6c00, 0xfe00, 'tpop%c\\t%O'),
  (16, 0xe800, 0xfe00, 'tadd%C\\t%0-2r, %3-5r, %6-8r'),
  (16, 0xea00, 0xfe00, 'tsub%C\\t%0-2r, %3-5r, %6-8r'),
  (16, 0xec00, 0xfe00, 'tadd%C\\t%0-2r, %3-5r, #%6-8d'),
  (16, 0xee00, 0xfe00, 'tsub%C\\t%0-2r, %3-5r, #%6-8d'),
  (16, 0x1200, 0xfe00, 'tstorerh%c\\t%0-2r, [%3-5r, %6-8r]'),
  (16, 0x1a00, 0xfe00, 'tloadrh%c\\t%0-2r, [%3-5r, %6-8r]'),
  (16, 0x1600, 0xf600, 'tloadrs%11?hb%c\\t%0-2r, [%3-5r, %6-8r]'),
  (16, 0x1000, 0xfa00, "tstorer%10'b%c\t%0-2r, [%3-5r, %6-8r]"),
  (16, 0x1800, 0xfa00, "tloadr%10'b%c\t%0-2r, [%3-5r, %6-8r]"),
  (16, 0xf000, 0xf800, 'tshftl%C\\t%0-2r, %3-5r, #%6-10d'),
  (16, 0xf800, 0xf800, 'tshftr%C\\t%0-2r, %3-5r, %s'),
  (16, 0xe000, 0xf800, 'tasr%C\\t%0-2r, %3-5r, %s'),
  (16, 0xa000, 0xf800, 'tmov%C\\t%8-10r, #%0-7d'),
  (16, 0xa800, 0xf800, 'tcmp%c\\t%8-10r, #%0-7d'),
  (16, 0xb000, 0xf800, 'tadd%C\\t%8-10r, #%0-7d'),
  (16, 0xb800, 0xf800, 'tsub%C\\t%8-10r, #%0-7d'),
  (16, 0x0800, 0xf800, 'tloadr%c\\t%8-10r, [pc, #%0-7W]\\t; (%0-7a)'),
  (16, 0x5000, 0xf800, 'tstorer%c\\t%0-2r, [%3-5r, #%6-10W]'),
  (16, 0x5800, 0xf800, 'tloadr%c\\t%0-2r, [%3-5r, #%6-10W]'),
  (16, 0x4000, 0xf800, 'tstorerb%c\\t%0-2r, [%3-5r, #%6-10d]'),
  (16, 0x4800, 0xf800, 'tloadrb%c\\t%0-2r, [%3-5r, #%6-10d]'),
  (16, 0x2000, 0xf800, 'tstorerh%c\\t%0-2r, [%3-5r, #%6-10H]'),
  (16, 0x2800, 0xf800, 'tloadrh%c\\t%0-2r, [%3-5r, #%6-10H]'),
  (16, 0x3000, 0xf800, 'tstorer%c\\t%8-10r, [sp, #%0-7W]'),
  (16, 0x3800, 0xf800, 'tloadr%c\\t%8-10r, [sp, #%0-7W]'),
  (16, 0x7000, 0xf800, 'tadd%c\\t%8-10r, pc, #%0-7W\\t; (t.add %8-10r, %0-7a)'),
  (16, 0x7800, 0xf800, 'tadd%c\\t%8-10r, sp, #%0-7W'),
  (16, 0xd000, 0xf800, 'tstorem%c\\t%8-10r!, %M'),
  (16, 0xd800, 0xf800, 'tloadm%c\\t%8-10r!, %M'),
  (16, 0xcf00, 0xff00, 'tserv%c\\t%0-7d'),
  (16, 0xc000, 0xf000, 'tj%8-11c.n\\t%0-7B%X'),
  (16, 0x8000, 0xf800, 'tj%c.n\\t%0-10B%x'),
  (32, 0x9000c000, 0xf800d000, 'tjlex%c\\t%B%x'),
  (32, 0x90009800, 0xf800f800, 'tjl%c\\t%B%x'),
]

class SLEIGHGenerator:
    def __init__(self):
        self.fields = set()
        self.register_fields = set()
        self.computed_definitions = {} # name -> definition string
        self.computed_registers = set() # Track which computed registers are used
        self.contexts = set()

    def generate(self):
        parsed_insns = []
        for size, value, mask, asm in insns:
            asm_clean = re.sub(r'\\t|\s+', ' ', asm).strip()
            asm_clean = re.sub(r'\s*;.*$', '', asm_clean)
            
            mnemonic, operands = self._parse_asm(asm_clean)
            constraints = self._build_constraints(size, value, mask)
            
            parsed_insns.append({
                'size': size,
                'mnemonic': mnemonic,
                'operands': operands,
                'constraints': constraints,
                'asm_clean': asm_clean,
                'original_asm': asm_clean # Use clean version for comment
            })
            
        self._print_headers()
        for insn in sorted(parsed_insns, key=lambda x: x['asm_clean']):
            self._print_instruction(insn)

    def _parse_asm(self, asm):
        parts = asm.split(' ', 1)
        mnemonic_raw = parts[0]
        operands_raw = parts[1] if len(parts) > 1 else ""
        
        # Clean mnemonic for P-code matching (strip control codes)
        mnemonic_clean = re.sub(r'%[\d\-\?\'\.]*[a-zA-Z]', '', mnemonic_raw)
        
        # Tokenize mnemonic by control codes
        mnem_parts = re.split(r'(%[\d\-\?\'\.]*[a-zA-Z]+)', mnemonic_raw)
        mnem_parts = [x for x in mnem_parts if x]
        
        # Tokenize operands
        op_parts = []
        if operands_raw:
            # Add space separator before operands if they exist
            op_parts.append(" ")
            # Split operands
            raw_ops = re.split(r'([,\[\]\s+#])', operands_raw)
            op_parts.extend([x for x in raw_ops if x and x.strip()])
            
        all_parts = mnem_parts + op_parts
        final_operands = []
        
        for part in all_parts:
            # If part contains %, it might be a control code or multiple control codes (e.g. %S%x)
            if '%' in part:
                 # Split by control code regex (same as mnemonic)
                 sub_parts = re.split(r'(%[\d\-\?\'\.]*[a-zA-Z]+)', part)
                 sub_parts = [x for x in sub_parts if x]
                 
                 for sub in sub_parts:
                     if '%' in sub:
                         expanded = self._expand_control_code(sub)
                         if expanded != '': 
                             final_operands.append(expanded)
                     else:
                         # Literal parts between codes (rare but possible)
                         if sub.strip() and not sub in [',', '[', ']', ' ', '+', '#']:
                             final_operands.append(f'"{sub}"')
                         elif sub.strip():
                             final_operands.append(sub)
            else:
                 # Check if literal identifier (needs quotes for display)
                 if part.strip() and not part in [',', '[', ']', ' ', '+', '#']:
                     final_operands.append(f'"{part}"')
                 else:
                     final_operands.append(part)
                 
        return mnemonic_clean, final_operands

    def _expand_control_code(self, code):
        # Handle prefix stripping if passed explicitly (legacy check, usually handled by split)
        prefix = ""
        if code.startswith("#"):
            prefix = "#"
            code = code[1:]
            
        # Register fields
        if code == '%0-2r': return prefix + self._reg_field(0, 3)
        if code == '%3-5r': return prefix + self._reg_field(3, 3)
        if code == '%6-8r': return prefix + self._reg_field(6, 3)
        if code == '%8-10r': return prefix + self._reg_field(8, 3)
        
        # High registers (computed)
        if code == '%D': 
            self._add_computed_reg('hi_reg_7_1_and_0_3', 'hi_reg_upper_7_1', 'hi_reg_lower_0_3')
            return prefix + 'hi_reg_7_1_and_0_3'
        if code == '%S': 
            self._add_computed_reg('hi_reg_6_1_and_3_3', 'hi_reg_upper_6_1', 'hi_reg_lower_3_3')
            return prefix + 'hi_reg_6_1_and_3_3'
        
        # Condition codes and Suffixes
        if code == '%c': return '' 
        if code == '%x': return '' # Ignore %x (hex format marker usually empty context)
        if code == '%X': return '' # Ignore %X
        if code == '%C': return '"s"' # Return quoted literal for display
        if code == '%8-11c': return self._field('cond_imm', 8, 4)
        
        # Immediates with scaling
        m = re.match(r'%(\d+)-(\d+)d', code)
        if m: return prefix + self._imm_field(int(m.group(1)), int(m.group(2)) - int(m.group(1)) + 1, 'dec')
        
        m = re.match(r'%(\d+)-(\d+)x', code)
        if m: return prefix + self._imm_field(int(m.group(1)), int(m.group(2)) - int(m.group(1)) + 1, 'hex')

        m = re.match(r'%(\d+)-(\d+)W', code)
        if m: 
            base = self._imm_field(int(m.group(1)), int(m.group(2)) - int(m.group(1)) + 1, 'dec')
            param = f'{base}_x4'
            self.computed_definitions[param] = f'{param}: value is {base} [ value = {base} * 4; ] {{ local tmp:4 = value; export tmp; }}'
            return prefix + param

        m = re.match(r'%(\d+)-(\d+)H', code)
        if m: 
            base = self._imm_field(int(m.group(1)), int(m.group(2)) - int(m.group(1)) + 1, 'dec')
            param = f'{base}_x2'
            self.computed_definitions[param] = f'{param}: value is {base} [ value = {base} * 2; ] {{ local tmp:4 = value; export tmp; }}'
            return prefix + param
        
        # Branches
        m = re.match(r'%(\d+)-(\d+)B', code)
        if m: return self._field(f'displacement_{m.group(1)}_{int(m.group(2))-int(m.group(1))+1}', int(m.group(1)), int(m.group(2))-int(m.group(1))+1, 'signed')
        
        if code == '%B': return 'rel22' # For 32-bit jumps

        # Register list literals - keep braces for display, extract name for constraints
        if code == '%M': return '{}'  # Empty register list
        if code == '%N': return '{lr}'  # lr register list
        if code == '%O': return '{pc}'  # pc register list
        if code == '%s': 
            self._add_shift_computed('right_shift_imm_6_5', 'imm_6_5')
            return 'right_shift_imm_6_5'
            
        # Bit flags
        m = re.match(r'%(\d+)\'b', code)
        if m: return self._field(f'flag_{m.group(1)}_1', int(m.group(1)), 1)
        
        m = re.match(r'%(\d+)\?hb', code)
        if m: return self._field(f'flag_{m.group(1)}_1', int(m.group(1)), 1)
            
        return prefix + code


    def _reg_field(self, lo, size):
        name = f'reg_{lo}_{size}'
        self.fields.add((name, lo, size, None))
        self.register_fields.add(name)
        return name

    def _imm_field(self, lo, size, attr):
        name = f'imm_{lo}_{size}'
        self.fields.add((name, lo, size, attr))
        return name

    def _field(self, name, lo, size, attr=None):
        self.fields.add((name, lo, size, attr))
        return name

    def _add_computed_reg(self, name, hi_field, lo_field):
        # We need the underlying fields
        hi_lo = int(hi_field.split('_')[-2])
        hi_sz = int(hi_field.split('_')[-1])
        lo_lo = int(lo_field.split('_')[-2])
        lo_sz = int(lo_field.split('_')[-1])
        
        self.fields.add((hi_field, hi_lo, hi_sz, None))
        self.fields.add((lo_field, lo_lo, lo_sz, None))
        
        # Track that this computed register is used
        self.computed_registers.add(name)
        
        # Add to computed defines
        def1 = f'{name}: {lo_field} is {hi_field}=1 & {lo_field} {{ local tmp:4 = {lo_field}; export tmp; }}'
        # Note: This logic is simplified; original had strict register mapping logic. 
        # For this version we will map these to specific registers later via attach variables
        self.computed_definitions[name] = f'# Computed register {name} needs manual attach variables logic'

    def _add_shift_computed(self, name, base_field):
        self._imm_field(6, 5, None) # Ensure base field exists
        self.computed_definitions[name] = f'{name}: 32 is {base_field}=0 {{ }}\n{name}: {base_field} is {base_field} {{ }}'

    def _build_constraints(self, size, value, mask):
        constraints = []
        # Same constraint building logic as before
        current_mask_chunk_start = -1
        for i in range(size):
            is_masked = (mask >> i) & 1
            if is_masked:
                if current_mask_chunk_start == -1: current_mask_chunk_start = i
            else:
                if current_mask_chunk_start != -1:
                    chunk_len = i - current_mask_chunk_start
                    chunk_val = (value >> current_mask_chunk_start) & ((1 << chunk_len) - 1)
                    op_name = f'op_{current_mask_chunk_start}_{chunk_len}' if size==16 else f'op32_{current_mask_chunk_start}_{chunk_len}'
                    self.fields.add((op_name, current_mask_chunk_start, chunk_len, None))
                    constraints.append(f'{op_name}=0x{chunk_val:x}')
                    current_mask_chunk_start = -1
        if current_mask_chunk_start != -1:
            chunk_len = size - current_mask_chunk_start
            chunk_val = (value >> current_mask_chunk_start) & ((1 << chunk_len) - 1)
            op_name = f'op_{current_mask_chunk_start}_{chunk_len}' if size==16 else f'op32_{current_mask_chunk_start}_{chunk_len}'
            self.fields.add((op_name, current_mask_chunk_start, chunk_len, None))
            constraints.append(f'{op_name}=0x{chunk_val:x}')
        return constraints

    def _print_headers(self):
        print('# Generated by generate_sleigh.py')
        print('define endian=little;')
        print('define alignment=2;')
        print('define space RAM type=ram_space size=4 default;')
        print('define space register type=register_space size=4;')
        print()
        print('define register offset=0x0000 size=4 [')
        print('  r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 sp lr pc')
        print('];')
        print()
        
        print('define token instr(16)')
        # Deduplicate fields: for same name, keep the one with attribute
        fields_dict = {}
        for name, lo, size, attr in self.fields:
            if name not in fields_dict or (attr and not fields_dict[name][3]):
                fields_dict[name] = (name, lo, size, attr)
        
        for name, lo, size, attr in sorted(fields_dict.values(), key=lambda x: (x[0], x[1], x[2], str(x[3]))):
            if not name.startswith('op32_'):
                attr_str = f' {attr}' if attr else ''
                print(f'  {name} = ({lo}, {lo + size - 1}){attr_str}')
        print(';')
        
        print('define token instr32(32)')
        for name, lo, size, attr in sorted(list(self.fields), key=lambda x: (x[0], x[1], x[2], str(x[3]))):
            if name.startswith('op32_'):
                attr_str = f' {attr}' if attr else ''
                print(f'  {name} = ({lo}, {lo + size - 1}){attr_str}')
        print('  imm22_hi = (16, 26)\n  imm22_lo = (0, 10)')
        print(';')
        print()
        
        # Computed register definitions (hi_reg_*)
        if 'hi_reg_7_1_and_0_3' in self.computed_registers or 'hi_reg_6_1_and_3_3' in self.computed_registers:
            if 'hi_reg_7_1_and_0_3' in self.computed_registers:
                print('hi_reg_7_1_and_0_3: reg is hi_reg_upper_7_1 & hi_reg_lower_0_3 [ reg = (hi_reg_upper_7_1 << 3) | hi_reg_lower_0_3; ] { local tmp:4 = reg; export tmp; }')
            if 'hi_reg_6_1_and_3_3' in self.computed_registers:
                print('hi_reg_6_1_and_3_3: reg is hi_reg_upper_6_1 & hi_reg_lower_3_3 [ reg = (hi_reg_upper_6_1 << 3) | hi_reg_lower_3_3; ] { local tmp:4 = reg; export tmp; }')
            print()
        
        # Computed values definitions
        for key in sorted(self.computed_definitions.keys()):
            if not key.startswith('hi_reg'):
                print(self.computed_definitions[key])
        print()
        
        # Rel definitions for jumps
        print('rel22: addr is imm22_hi & imm22_lo [ addr = ((imm22_hi << 11) | imm22_lo) << 1; ] { local tmp:4 = inst_next + addr; export tmp; }')
        print()

        # Attach variables
        if self.register_fields:
            # We assume all reg_X_3 map to r0-r7
            reg_fields_list = ' '.join(sorted(list(self.register_fields)))
            print(f'attach variables [ {reg_fields_list} ] [ r0 r1 r2 r3 r4 r5 r6 r7 ];')
        
        # Manually attach high registers based on known logic
        print('attach variables [ hi_reg_lower_0_3 hi_reg_lower_3_3 ] [ r8 r9 r10 r11 r12 sp lr pc ];')
        print()

    def _print_instruction(self, insn):
        print(f'# {insn["original_asm"]}')
        
        display_str = ""
        prev_is_ident = False
        parsed_ops = insn['operands']
        
        # Clean up empty strings
        parsed_ops = [x for x in parsed_ops if x != '']

        for part in parsed_ops:
            is_ident = not part.startswith('"') and not part in [',', ' ', '[', ']', '+', '#']
            if prev_is_ident and (is_ident or part.startswith('"')):
                display_str += "^"
            display_str += part
            if is_ident or part.startswith('"'):
                prev_is_ident = True
            else:
                prev_is_ident = False
                
        constraint_str = ' & '.join(insn['constraints'])
        semantic_ops = self._get_semantic_ops(parsed_ops)
        if semantic_ops:
             constraint_str += ' & ' + ' & '.join(semantic_ops)
             
        print(f':{display_str} is {constraint_str}')
        print('{')
        self._print_pcode(insn['mnemonic'], semantic_ops)
        print('}')
        print()

    def _get_semantic_ops(self, ops):
        # Filter out formatting for constraints
        res = []
        for x in ops:
            # Skip quoted literals and punctuation
            if x.startswith('"') or x in [',', ' ', '[', ']', '{', '}', '+', '#']: continue
            # Handle register list literals: {pc}, {lr}, {}
            if x.startswith('{') and x.endswith('}'):
                # Extract register name from {reg} for constraint
                inner = x[1:-1]  # Remove braces
                if inner and inner not in ['']:  # Skip empty {}
                    res.append(inner)  # Add bare register name (pc, lr, etc.)
                continue
            # Skip global registers (exact match) when not in braces
            if x in ['sp', 'pc']: continue
            res.append(x)
        return res

    def _print_pcode(self, mnem, ops):
        # Clean ops
        clean_ops = [op.replace('#', '') for op in ops]
        
        if mnem.startswith('tmov'):
            if len(clean_ops) >= 2: print(f'    {clean_ops[0]} = {clean_ops[1]};')
        elif mnem.startswith('tadd'):
            if len(clean_ops) >= 2:
                if len(clean_ops) >= 3: print(f'    {clean_ops[0]} = {clean_ops[1]} + {clean_ops[2]};')
                else: print(f'    {clean_ops[0]} = {clean_ops[0]} + {clean_ops[1]};')
        elif mnem.startswith('tsub'):
            if len(clean_ops) >= 3: print(f'    {clean_ops[0]} = {clean_ops[1]} - {clean_ops[2]};')
            elif len(clean_ops) >= 2: print(f'    {clean_ops[0]} = {clean_ops[0]} - {clean_ops[1]};')
        elif mnem.startswith('tmul'):
             if len(clean_ops) >= 2: print(f'    {clean_ops[0]} = {clean_ops[0]} * {clean_ops[1]};')
        elif mnem.startswith('tand'):
             if len(clean_ops) >= 2: print(f'    {clean_ops[0]} = {clean_ops[0]} & {clean_ops[1]};')
        elif mnem.startswith('tor'):
             if len(clean_ops) >= 2: print(f'    {clean_ops[0]} = {clean_ops[0]} | {clean_ops[1]};')
        elif mnem.startswith('txor'):
             if len(clean_ops) >= 2: print(f'    {clean_ops[0]} = {clean_ops[0]} ^ {clean_ops[1]};')
        elif mnem.startswith('tshftl'):
             if len(clean_ops) >= 2: print(f'    {clean_ops[0]} = {clean_ops[0]} << {clean_ops[1]};')
        elif mnem.startswith('tshftr'):
             if len(clean_ops) >= 2: print(f'    {clean_ops[0]} = {clean_ops[0]} >> {clean_ops[1]};')
        elif mnem.startswith('tasr'):
             if len(clean_ops) >= 2: print(f'    {clean_ops[0]} = {clean_ops[0]} s>> {clean_ops[1]};')
        elif mnem.startswith('tj'):
             if 'l' not in mnem and 'ex' not in mnem and clean_ops:
                 # For conditional jumps like tj<cond>.n, the displacement is usually the last operand
                 # Find displacement field (starts with 'displacement_')
                 disp = next((op for op in clean_ops if 'displacement' in op), None)
                 if disp:
                     print(f'    # Conditional jump - TODO: implement condition check for {clean_ops[0] if clean_ops else "cond"}')
                     print(f'    # goto {disp};')
                 elif clean_ops:
                     # Fallback to first operand if no displacement found
                     print(f'    # TODO: Jump logic for {clean_ops[0]}')
             else:
                 print(f'    # TODO: Complex jump logic')
        else:
            print(f'    # TODO: P-code for {mnem}')

if __name__ == '__main__':
    gen = SLEIGHGenerator()
    gen.generate()
