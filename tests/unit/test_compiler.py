import yaml
from constraint_lattice.compiler import ConstraintOntologyCompiler

def test_compile_file(tmp_path):
    data = {
        'constraints': [
            {'name': 'test', 'type': 'text', 'params': {'foo': 'bar'}}
        ]
    }
    path = tmp_path / 'c.yaml'
    path.write_text(yaml.safe_dump(data))
    compiler = ConstraintOntologyCompiler()
    compiled = compiler.compile_file(str(path))
    assert len(compiled) == 1
    assert compiled[0].name == 'test'
    assert compiled[0].type == 'text'
