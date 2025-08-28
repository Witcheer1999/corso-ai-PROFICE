from crewai.tools import tool
import ast
import math
import re

ALLOWED_MATH_FUNCTIONS = {
    "abs", "round", "min", "max", "sum", "pow", "int", "float", "complex",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "sqrt", "exp", "log", "log10", "log2",
    "ceil", "floor", "degrees", "radians", "factorial", "pi", "e"
}

FORBIDDEN_PATTERNS = [
    r'import\s+(?!math\b)', r'from\s+(?!math\b)', r'__.*__', r'exec\s*\(',
    r'eval\s*\(', r'open\s*\(', r'input\s*\(', r'print\s*\(', r'os\.',
    r'sys\.', r'subprocess', r'socket', r'file\s*\(', r'globals\s*\('
]

@tool("Math function executor")
def execute_math_function(function_code: str) -> str:
    """
    Esegue SOLO funzioni matematiche in Python.
    Input: codice Python con una singola funzione matematica.
    Output: risultato numerico dell'esecuzione.
    """
    # Guardrail INPUT
    if not isinstance(function_code, str) or len(function_code.strip()) == 0:
        raise ValueError("Codice vuoto o non valido.")
    
    if len(function_code) > 2000:
        raise ValueError("Codice troppo lungo (max 2000 caratteri).")
    
    # Controlla pattern pericolosi
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, function_code, re.IGNORECASE):
            raise ValueError("Codice contiene elementi non matematici non consentiti.")
    
    # Parse AST e valida nomi
    try:
        tree = ast.parse(function_code)
    except SyntaxError:
        raise ValueError("Sintassi Python non valida.")
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_MATH_FUNCTIONS and not node.id.startswith(('x', 'y', 'z', 'a', 'b', 'c', 'n', 't', 'def', 'return')):
                raise ValueError(f"Nome non consentito in funzioni matematiche: {node.id}")
    
    # Controlla che ci sia esattamente una funzione
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise ValueError("Deve essere definita esattamente una funzione.")
    
    try:
        # Ambiente sicuro per esecuzione
        safe_globals = {
            '__builtins__': {},
            'math': math,
            'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum, 'pow': pow,
            'int': int, 'float': float, 'complex': complex,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
            'pi': math.pi, 'e': math.e
        }
        
        # Esegui il codice
        exec(function_code, safe_globals)
        
        # Trova la funzione definita
        func_name = functions[0].name
        if func_name not in safe_globals:
            raise ValueError("Funzione non trovata dopo l'esecuzione.")
        
        func = safe_globals[func_name]
        result = func()
        
    except Exception as e:
        raise ValueError(f"Errore durante l'esecuzione: {str(e)}")
    
    # Guardrail OUTPUT
    if not isinstance(result, (int, float, complex)):
        raise ValueError("La funzione deve restituire un valore numerico.")
    
    if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
        raise ValueError("Risultato non valido (NaN o infinito).")
    
    return f"Risultato: {result}"