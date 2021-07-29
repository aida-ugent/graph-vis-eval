import time
import os
import logging
import numpy as np

from evalne.evaluation.evaluator import LPEvaluator
from evalne.utils import util
from evalne.utils import preprocess as pp

class GravisLPEvaluator(LPEvaluator):
    
    def _evaluate_ne_cmd(self, data_split, method_name, command, edge_embedding_methods,
                         input_delim, output_delim, write_weights, write_dir, timeout, verbose,
                         ne_postprocessing_fn, embedding_dim):
        """
        The actual implementation of the node embedding evaluation. Stores the train graph as an edgelist to a
        temporal file and provides it as input to the method evaluated. Performs the command line call and reads
        the output. Node embeddings are transformed to node-pair embeddings and predictions are run.

        Returns
        -------
        results : list
            A list of results, one for each node-pair embedding method set.
        """
        # Create temporal files with in/out data for method
        tmpedg = './edgelist.tmp'
        tmpemb = './emb.tmp'

        # Write the train data to a file
        data_split.save_tr_graph(tmpedg, delimiter=input_delim, write_stats=False,
                                 write_weights=write_weights, write_dir=write_dir)

        # Add the input, output and embedding dimensionality to the command
        command = command.format(tmpedg, tmpemb, embedding_dim)

        print('Running command...')
        print(command)

        try:
            # Call the method
            util.run(command, timeout, verbose)

            # Some methods append a .txt filetype to the outfile if its the case, read the txt
            if os.path.isfile('./emb.tmp.txt'):
                tmpemb = './emb.tmp.txt'

            # Read embeddings from output file
            X = pp.read_node_embeddings(tmpemb, data_split.TG.nodes, embedding_dim, output_delim, method_name)
            prev_dims = X['0'].shape[0]

            # Postprocessing
            if ne_postprocessing_fn is not None:
                X = dict(zip(list(X.keys()), ne_postprocessing_fn(np.asarray(list(X.values())))))
                print(f"Post-processed the embedding from {prev_dims} to {X['0'].shape[0]} dimensions.")
            
            # Check that embeddings have specified dimensions
            assert X['0'].shape[0] == self.dim, f"Embedding is expected to have {self.dim} dimensions but has {X['0'].shape[0]}."    
                
            # Evaluate the model
            results = list()
            for ee in edge_embedding_methods:
                results.append(self.evaluate_ne(data_split=data_split, X=X, method=method_name, edge_embed_method=ee))
            return results

        except (IOError, OSError):
            raise IOError('Execution of method `{}` did not generate node embeddings file. \nPossible reasons: '
                          '1) method is not correctly installed or 2) wrong method call or parameters... '
                          '\nSetting verbose=True can provide more information.'.format(method_name))

        finally:
            # Delete the temporal files
            if os.path.isfile(tmpedg):
                os.remove(tmpedg)
            if os.path.isfile(tmpemb):
                os.remove(tmpemb)
            if os.path.isfile('./emb.tmp.txt'):
                os.remove('./emb.tmp.txt')
    

    def evaluate_cmd(self, method_name, method_type, command, edge_embedding_methods, input_delim, output_delim,
                     tune_params=None, maximize='auroc', write_weights=False, write_dir=False,
                     timeout=None, verbose=True, ne_postprocessing_fn=None, embedding_dim=None):
        
        # Measure execution time
        start = time.time()
        if timeout is None:
            timeout = 31536000
            
        # embedding_dim can be different from default if we have postprocessing with t-SNE
        if embedding_dim is None:
            embedding_dim = self.dim

        # Check if a validation set needs to be initialized
        if self.trainvalid_split is None or len(self.trainvalid_split.test_edges) == 0:
            self._init_trainvalid()

        # Check the method type and raise an error if necessary
        if method_type not in ['ne', 'ee', 'e2e']:
            raise ValueError('Method type `{}` of method `{}` is unknown! Valid options are: `ne`, `ee`, `e2e`'
                                .format(method_type, method_name))

        # If the method evaluated does not require node-pair embeddings set this parameter to ['none']
        if method_type != 'ne':
            edge_embedding_methods = ['none']
            self.edge_embed_method = None
            
        # Call the corresponding evaluation method
        if method_type == 'ee' or method_type == 'e2e':
            results = self._evaluate_ee_e2e_cmd(self.traintest_split, method_name, method_type, command,
                                                input_delim, output_delim, write_weights, write_dir,
                                                timeout - (time.time() - start), verbose)
        else:
            # We still have to tune the node-pair embedding method
            if len(edge_embedding_methods) > 1:
                # For NE methods first compute the results on validation data
                valid_results = self._evaluate_ne_cmd(self.trainvalid_split, method_name, command,
                                                        edge_embedding_methods, input_delim, output_delim,
                                                        write_weights, write_dir, timeout-(time.time()-start),
                                                        verbose=False, 
                                                        ne_postprocessing_fn=ne_postprocessing_fn,
                                                        embedding_dim=embedding_dim)

                # Extract and log the validation scores
                ee_scores = list()
                for i in range(len(valid_results)):
                    func = getattr(valid_results[i].test_scores, str(maximize))
                    bestscore = func()
                    ee_scores.append(bestscore)
                    logging.info('Validation score for method `{}_{}` is: {}, no other tuned params.'
                                    .format(method_name, edge_embedding_methods[i], bestscore))

                # We now select the ee that performs best in terms of maximize score
                best_ee_idx = int(np.argmax(ee_scores))
            else:
                # If we only have one ee method then that the one we compute results for, no need for validation
                best_ee_idx = 0

            # Compute the results on the full train split
            results = self._evaluate_ne_cmd(self.traintest_split, method_name, command,
                                            [edge_embedding_methods[best_ee_idx]], input_delim, output_delim,
                                            write_weights, write_dir, timeout, verbose,
                                            ne_postprocessing_fn=ne_postprocessing_fn,
                                            embedding_dim=embedding_dim)

        # End of exec time measurement
        end = time.time() - start
        res = results[0]
        res.params.update({'eval_time': end})

        # Return the evaluation results
        return res